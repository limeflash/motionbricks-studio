"""MotionBricks worker: the motion player on our own authored poses (transitions, hit reactions, knock-backs).
Run by the Go server with H:\\tools\\motionbricks\\.venv\\Scripts\\python.exe, cwd H:\\tools\\motionbricks. Ported from the
probe's mb_generate.py (seq_take, the hit recipes). Never touches the BONES sample clips: the skeleton comes from the
asset registry, every target pose from soma.py's library or the caller's spec."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto  # noqa: E402  (first: it moves library prints off stdout)
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import soma as S  # noqa: E402

REPO = os.environ.get("MBS_MOTIONBRICKS", r"H:\tools\motionbricks")
wp.config.kernel_cache_dir = os.environ.get("MBS_WARP_CACHE", r"H:\tools\warp-cache")
wp.config.quiet = True
sys.path.insert(0, REPO)
wp.init()
wp.set_device("cuda:0")
from motionbricks.model.contract import ModelRig  # noqa: E402
from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH  # noqa: E402
from motionbricks.model.soma.engine import MotionBricksSOMAEngine  # noqa: E402
from motionbricks.runtime.motion_player import GeneratedMotionPlayer  # noqa: E402
from motionbricks.runtime.target_pose import TargetPoseInferenceParams  # noqa: E402
from app.runtime.common.rig import load_rig_assets, model_to_runtime  # noqa: E402

SK = load_rig_assets(ModelRig.SOMA).clip_registry.skeleton
_M = model_to_runtime(ModelRig.SOMA)
SKEL = os.environ["MBS_SKELETON"]
if not os.path.exists(SKEL):
    os.makedirs(os.path.dirname(SKEL), exist_ok=True)
    np.savez(SKEL, names=np.array(list(SK.joint_names)), parents=np.array([SK.joint_parent(i) for i in range(SK.num_joints)]),
             ref=SK.reference_local_transforms.numpy().astype(np.float64), m=np.array([*_M.p, *_M.q], np.float64))
S.load_skeleton(SKEL)
t0 = time.perf_counter()
engine = MotionBricksSOMAEngine(SOMA_N1_FP32_ONNX_PATH, model_to_runtime=_M)
ENGINE_LOAD_S = round(time.perf_counter() - t0, 2)
GEN = []
_gen = engine.generate_animation


def _timed(*a, **k):
    engine.synchronize()
    t = time.perf_counter()
    r = _gen(*a, **k)
    engine.synchronize()
    GEN.append(time.perf_counter() - t)
    return r


engine.generate_animation = _timed


def wp_poses(poses):
    return wp.array(np.stack(poses).astype(np.float32), dtype=wp.transform)


def tp(tokens=None, n=1, early=0):
    return TargetPoseInferenceParams(num_poses_to_use=n, transition_early_by_frames=early,
                                     allowed_token_counts=tuple(tokens) if tokens else None)


def seq_take(start, context, steps, hold=12, lead=10):
    """the motion player on world poses: `lead` frames of the start pose, the context frames recorded as shown, then
    one generation per step (targets, params) played to its transition point; the last held `hold` frames. Returns
    poses, foot contacts and the frame each step's target is reached"""
    player = GeneratedMotionPlayer(engine, SK)
    cur = wp.array(start.astype(np.float32), dtype=wp.transform)
    player.seed_history_from_pose(cur)
    fcw = wp.zeros(4, dtype=wp.float32)
    poses, fc, marks = [start] * lead, [np.ones(4)] * lead, []
    for fr in context:
        player.record_pose(S.DT, wp.array(fr.astype(np.float32), dtype=wp.transform))
        poses.append(fr)
        fc.append(np.ones(4))
    for targets, params in steps:
        player.generate(wp_poses(targets), params)
        while True:
            player.sample(S.DT, cur, fcw)
            player.record_pose(S.DT, cur)
            poses.append(cur.numpy().astype(np.float64))
            fc.append(fcw.numpy().astype(np.float64))
            if player.time_to_transition <= 1e-6:
                break
        marks.append(len(poses) - 1)
    poses += [poses[-1]] * hold
    fc += [fc[-1]] * hold
    return poses, fc, marks


def align_to(model_pose, end_world):
    """a model-space pose stood where the motion ends: its hips' ground spot and heading set to the end pose's"""
    end = S.from_world(end_world)
    yaw = S.heading(end) - S.heading(model_pose)
    p = S.place(model_pose, yaw=yaw)
    p[0, [0, 2]] = end[0, [0, 2]]
    return p


def target(spec, end_world):
    """a step target: a pose spec, or {"take": dir, "frame": n} (a frame of another take, stood where we are; its
    "root" moves it on from there)"""
    if isinstance(spec, dict) and "take" in spec:
        z = np.load(os.path.join(spec["take"], "take.npz"))["poses"]
        p = align_to(z[int(spec.get("frame", 0))], end_world)
        r = spec.get("root") or {}
        return S.to_world(S.place(p, r.get("forward", 0.0), r.get("left", 0.0), r.get("up", 0.0), r.get("yaw", 0.0)))
    return S.to_world(S.build(spec))


def run(out, start, context, steps, hold, lead, meta):
    n0 = len(GEN)
    with wp.ScopedStream(engine.stream):
        poses, fc, marks = seq_take(start, context, steps, hold, lead)
        engine.synchronize()
    g = GEN[n0:]
    meta = dict(meta, engine="MotionBricks SOMA N1 FP32 (TensorRT)", generations=len(g),
                gen_ms=[round(x * 1000, 2) for x in g], step_frames=marks)
    return S.save_take(out, poses, fc, meta)


def sequence(out, start=None, from_take=None, keep_from=None, context=(), steps=(), hold=12, lead=10):
    """transitions and inbetweens: from a pose (spec) or the end of a take, through each step's targets"""
    ctx = [S.to_world(S.build(c)) for c in context]
    if from_take:
        prev, _ = S.load_take(from_take)
        keep = len(prev) if keep_from is None else max(1, int(keep_from))
        st, ctx, lead = prev[-keep], prev[-keep + 1:] + ctx, 1
    else:
        st = S.to_world(S.build(start or "ready"))
    built, end = [], (ctx[-1] if ctx else st)
    for s in steps:
        tg = [target(t, end) for t in s["targets"]]
        built.append((tg, tp(s.get("tokens"), len(tg), s.get("early", 0))))
        end = tg[-1]
    return run(out, st, ctx, built, hold, lead, {"op": "sequence"})


# the hit recipes of the probe (2026-10-27): travel away from the blow, scaled to `distance` (m on SOMA)
AWAY = {"front": (0.0, -1.0), "back": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}   # (left, forward)
RECIPES = {
    "light": (0.35, [], [(("hit", 0.12), (6,)), (("ready", 0.35), range(8, 13))]),
    "medium": (0.42, [(0.3, 0.02), (0.65, 0.06), (0.9, 0.11), (1.0, 0.16)], [(("ready", 0.42), range(8, 14))]),
    "knockback": (0.9, [(0.3, 0.04), (0.65, 0.13), (0.9, 0.24), (1.0, 0.36)],
                  [(("crouch_land", 0.75), (6, 7, 8)), (("ready", 0.9), range(8, 14))]),
}


def hit_reaction(out, direction="front", strength="knockback", distance=None, land="crouch", hold=12, lead=10):
    base, impact, plan = RECIPES[strength]
    k = 1.0 if distance is None else float(distance) / base
    ax, az = AWAY[direction]
    ready, hit = S.build("ready"), S.build("hit_" + direction)

    def moved(p, d):
        return S.place(p, forward=az * d * k, left=ax * d * k)

    ctx = []
    for w, d in impact:                             # the blow already landing: the context the model continues from
        b = S.blend(ready, hit, w)
        ctx.append(S.to_world(moved(b, d)))
    steps = []
    for (name, d), tokens in plan:
        if name == "crouch_land" and land != "crouch":
            continue
        p = hit if name == "hit" else S.build(name)
        steps.append(([S.to_world(moved(p, d))], tp(list(tokens))))
    meta = {"op": "hit_reaction", "direction": direction, "strength": strength,
            "distance_m": round(base * k, 3), "land": land, "impact_frame": lead}
    return run(out, S.to_world(ready), ctx, steps, hold, lead, meta)


def poses(names=None):
    """the library with each pose's description and a few joint positions (model space), for the caller"""
    out = {}
    for n, (spec, desc) in S.POSES.items():
        W = S.fk(S.build(n))
        out[n] = {"spec": spec, "about": desc, "joints": {j: W[S.JI[j], :3].round(3).tolist() for j in (
            "Hips", "Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot")}}
    return {"poses": out}


proto.serve({"sequence": sequence, "hit_reaction": hit_reaction, "poses": poses},
            {"worker": "motionbricks", "engine_load_s": ENGINE_LOAD_S, "joints": len(S.NAMES)})
