"""Kimodo worker, text-free: motion steered by full-body key poses, end-effector targets and the root's ground path.
Run by the Go server with H:\\tools\\kimodo\\.venv\\Scripts\\python.exe (HF_HOME=H:\\tools\\hf, offline). Ported from the
probe's kimodo_probe.py (move) and mb_generate.py (kimodo_motion). Llama / LLM2Vec are never loaded: NoTextEncoder
stands in and only empty prompts pass (Kimodo zeroes an empty prompt's text features: its unconditional branch)."""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto  # noqa: E402  (first: it moves library prints off stdout)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import soma as S  # noqa: E402

MODEL = "Kimodo-SOMA-RP-v1.2"
DEV = "cuda:0"
S.load_skeleton(os.environ["MBS_SKELETON"])        # written by the MotionBricks worker (the Go server runs it first)


class NoTextEncoder:
    llm_dim = 4096

    def __call__(self, texts):
        assert all(not t.strip() for t in texts), "a text prompt needs the gated text encoder, which is cut out"
        return torch.zeros(len(texts), 1, self.llm_dim), [0] * len(texts)

    def to(self, device):
        return self

    def eval(self):
        return self


from kimodo.constraints import EndEffectorConstraintSet, FullBodyConstraintSet, Root2DConstraintSet  # noqa: E402
from kimodo.model.load_model import load_model  # noqa: E402
from kimodo.skeleton.definitions import SOMASkeleton77  # noqa: E402

t0 = time.perf_counter()
model = load_model(MODEL, device=DEV, text_encoder=NoTextEncoder())
LOAD_S = round(time.perf_counter() - t0, 2)
SK77 = SOMASkeleton77()
K_NAMES = list(SK77.bone_order_names)
K_PAR = [K_NAMES.index(p) if p else -1 for _, p in SK77.bone_order_names_with_parents]
G0 = SK77.fk(torch.eye(3).repeat(1, SK77.nbjoints, 1, 1), torch.zeros(1, 3))[0][0].numpy()   # Kimodo's T-pose
MB_OF_K = [S.JI[n] for n in K_NAMES]               # Kimodo joint -> MotionBricks joint, by name
K_OF_MB = [K_NAMES.index(n) for n in S.NAMES]
WREF = [R.from_quat(q) for q in S.fk(S.REF)[:, 3:]]


def skeleton_poses(poses):
    """model-space SOMA poses -> the model skeleton's global rotations and joint positions (CUDA), as kimodo_probe did:
    each joint's world turn from the MotionBricks T-pose laid on Kimodo's T-pose"""
    D = np.array([[(R.from_quat(S.fk(p)[j, 3:]) * WREF[j].inv()).as_matrix() for j in MB_OF_K] for p in poses])
    Gk = np.einsum("kjab,jbc->kjac", D, G0)
    loc = Gk.copy()
    for j, p in enumerate(K_PAR):
        if p >= 0:
            loc[:, j] = np.transpose(Gk[:, p], (0, 2, 1)) @ Gk[:, j]
    hips = np.array([S.fk(p)[0, :3] for p in poses])
    sk = model.skeleton
    loc_t = torch.tensor(loc, dtype=torch.float32, device=DEV)
    if sk.nbjoints != 77:
        loc_t = sk.from_SOMASkeleton77(loc_t)
    Gc, Pc, _ = sk.fk(loc_t, torch.tensor(hips, dtype=torch.float32, device=DEV))
    return sk, Gc, Pc


def to_poses(out):
    """Kimodo's output -> MotionBricks world poses and foot contacts (L heel, L toe, R heel, R toe)"""
    G, root = out["global_rot_mats"][0], out["root_positions"][0]
    poses = []
    for t in range(len(G)):
        W = [R.from_matrix(G[t, K_OF_MB[j]]) * WREF[j] for j in range(len(S.NAMES))]
        p = S.REF.copy()
        for j in range(len(S.NAMES)):
            p[j, 3:] = ((W[S.PAR[j]].inv() if S.PAR[j] >= 0 else R.identity()) * W[j]).as_quat()
        p[0, :3] = root[t]
        poses.append(S.to_world(p))
    c = out["foot_contacts"][0].astype(float)
    if c.shape[1] == 6:
        c = np.stack([c[:, 0], c[:, 1:3].max(1), c[:, 3], c[:, 4:6].max(1)], 1)
    return poses, c


def solve(out_dir, n, keys, effectors=(), root_path=None, seed=0, steps=100, meta=None, post=None):
    """keys: [(frame, pose)], effectors: [(joint, frame, pose with the joint on its target)], root_path: [T, 3] rows of
    (frame, x, z) in model space or None; post(model-space poses) adjusts the result (a recipe's own touch-up)"""
    if not keys and root_path is None:
        raise ValueError("give at least one key or a root path")
    if not 8 <= n <= 300:
        raise ValueError("frames must be 8..300")
    cons, report = [], []
    if keys:
        keys = sorted(keys, key=lambda k: k[0])
        frames = [int(f) for f, _ in keys]
        if len(set(frames)) != len(frames) or frames[0] < 0 or frames[-1] >= n:
            raise ValueError(f"key frames must be distinct and within 0..{n - 1}: {frames}")
        sk, Gc, Pc = skeleton_poses([p for _, p in keys])
        cons.append(FullBodyConstraintSet(sk, torch.tensor(frames, device=DEV), Pc, Gc))
    for joint, f, pose in effectors:
        sk, Ge, Pe = skeleton_poses([pose])
        c = EndEffectorConstraintSet(sk, torch.tensor([int(f)], device=DEV), Pe, Ge, None, joint_names=[joint])
        c.pos_indices, c.rot_indices = c.pos_indices.to(DEV), c.rot_indices.to(DEV)   # built on the CPU by Kimodo
        cons.append(c)
    if root_path is not None:
        rp = np.asarray(root_path, float)
        f = np.arange(int(rp[0, 0]), int(rp[-1, 0]) + 1)
        xz = np.stack([np.interp(f, rp[:, 0], rp[:, 1]), np.interp(f, rp[:, 0], rp[:, 2])], 1)
        cons.append(Root2DConstraintSet(model.skeleton, torch.tensor(f, device=DEV),
                                        torch.tensor(xz, dtype=torch.float32, device=DEV)))
    torch.manual_seed(int(seed))
    t = time.perf_counter()
    out = model([""], [int(n)], int(steps), constraint_lst=cons, num_samples=1, return_numpy=True,
                post_processing=False)
    gen = time.perf_counter() - t
    if keys:                                       # how close the keys came out
        pos = out["posed_joints"][0]
        sub = [K_NAMES.index(nm) for nm, _ in model.skeleton.bone_order_names_with_parents] \
            if pos.shape[1] == 77 else list(range(pos.shape[1]))
        err = np.linalg.norm(pos[frames][:, sub] - Pc.cpu().numpy(), axis=-1)
        report = [round(100 * float(e), 1) for e in err.max(1)]
    poses, fc = to_poses(out)
    if post is not None:                           # raising constraint guidance only broke the poses (tried 4-9)
        poses = [S.to_world(p) for p in post([S.from_world(p) for p in poses])]
    meta = dict(meta or {}, engine=f"Kimodo {MODEL} (text-free)", gen_s=round(gen, 2), seed=int(seed),
                diffusion_steps=int(steps), key_frames=[int(k[0]) for k in keys], key_max_miss_cm=report,
                peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    return S.save_take(out_dir, poses, fc, meta)


def hips_xz(p):
    return S.fk(p)[0, [0, 2]]


def generate(out, n, keys, effectors=(), root_path="keys", seed=0, steps=100):
    """the generic tool: keys [{frame, pose}], effectors [{joint, frame, position, pose?}], root_path "keys" | "none" |
    [[frame, left, forward], ...]"""
    built = [(int(k["frame"]), S.build(k["pose"])) for k in keys]
    eff = []
    for e in effectors:
        f = int(e["frame"])
        if e.get("pose") is not None:
            p = S.build(e["pose"])
        else:                                       # the nearest key's pose, as the body around the target
            if not built:
                raise ValueError("an effector needs a pose when there are no keys")
            p = min(built, key=lambda k: abs(k[0] - f))[1].copy()
        up, lo = S.CHAIN[e["joint"]]
        S.two_bone(p, up, lo, e["joint"], e["position"], (0, -1, 0.3) if "Hand" in e["joint"] else (0, 0, 1))
        eff.append((e["joint"], f, p))
    rp = None
    if isinstance(root_path, list):
        rp = [(f, x, z) for f, x, z in root_path]
    elif root_path == "keys" and len(built) > 1:
        rp = [(f, *hips_xz(p)) for f, p in sorted(built, key=lambda k: k[0])]
    return solve(out, n, built, eff, rp, seed, steps, {"op": "generate"})


# ---- recipes ---------------------------------------------------------------------------------------------------------
def at(pose, x=0.0, z=0.0, yaw=0.0, hips_y=None):
    """a pose stood at (x, z) facing yaw (deg left of +Z), its hips raised to hips_y if given"""
    p = S.place(pose, yaw=yaw)
    p[0, 0] += x
    p[0, 2] += z
    if hips_y is not None:
        p[0, 1] += hips_y - S.fk(p)[0, 1]
    return p


def vault(out, height=0.64, speed="run", approach=1.53, seed=0, lift=0.6, clear=0.8, plant_y=0.45):
    """over an obstacle `height` m tall whose near face is `approach` m ahead: the run-up on the root path, our own keys
    for the hands planted on the top, the knees driven up past the near edge (`lift`: hips over the top), the legs
    tucked over it (`clear`), the landing crouch, and the run-out. Kimodo lets the palms drift up with the body, so
    they are laid back on the top afterwards (plant_hands)"""
    run = speed == "run"
    v = 3.4 if run else 1.5                          # m/s on SOMA
    h, a = float(height), float(approach)
    depth = 0.35                                     # the obstacle's depth (m)
    f_plant = max(12, int(round((a - 0.3) / v * 30)))
    f_lift = f_plant + (4 if run else 6)
    f_over = f_lift + (5 if run else 7)
    f_land = f_over + (9 if run else 12)
    f_end = f_land + 24

    def hands(p, z):                                 # both hands planted on the top
        for s, x in (("LeftHand", 0.18), ("RightHand", -0.18)):
            S.two_bone(p, *S.CHAIN[s], s, (x, h + 0.04, z), (0.4 if x > 0 else -0.4, 0.3, -1))
        return p

    # the plant: low and pitched well forward, so the shoulders come over the hands and the top is in reach
    reach = {"base": "ready", "legs": [40, 45, 35], "turns": {"Hips": [["x", 35]], "Spine2": [["x", 15]],
                                                              "Chest": [["x", 20]], "Head": [["x", -30]]}}
    plant = hands(at(S.build(reach), z=a - 0.38, hips_y=min(0.85, h * 0.5 + plant_y)), a + 0.06)
    tuck = {"base": "ready", "legs": [85, 115, 10], "turns": {"Hips": [["x", 10]], "Chest": [["x", 18]],
                                                               "Head": [["x", -22]]}}
    up = hands(at(S.build(tuck), z=a - 0.12, hips_y=h + lift), a + 0.06)
    over = hands(at(S.build(tuck), z=a + depth * 0.5 + 0.05, hips_y=h + clear), a + 0.1)
    land = at(S.build("crouch_land"), z=a + depth + 0.75)
    end = at(S.build("ready"), z=a + depth + 0.75 + (2.0 if run else 1.0))
    keys = [(f_plant, plant), (f_lift, up), (f_over, over), (f_land, land), (f_end, end)]

    def plant_hands(poses):
        """the palms eased onto the top over 4 frames before the plant, held there to the lift key, eased off by the
        over key (smoothstep weights: no snap); each arm re-solved around its own elbow, the hand keeping its turn.
        Where the top is out of reach the arm reaches for it: the push-off"""
        out = list(poses)
        for f in range(f_plant - 4, min(f_over, len(poses) - 1) + 1):
            u = (f - f_plant + 4) / 4 if f < f_plant else 1.0 if f <= f_lift else 1 - (f - f_lift) / (f_over - f_lift)
            w = u * u * (3 - 2 * u)
            p = out[f].copy()
            for s, x in (("Left", 0.18), ("Right", -0.18)):
                W = S.fk(p)
                cur = W[S.JI[s + "Hand"], :3]
                goal = cur + w * (np.array([W[0, 0] + x, h + 0.04, a + 0.06]) - cur)
                mid = (W[S.JI[s + "Arm"], :3] + cur) / 2
                S.two_bone(p, s + "Arm", s + "ForeArm", s + "Hand", goal, W[S.JI[s + "ForeArm"], :3] - mid)
            out[f] = p
        return out

    rp = [(0, 0.0, 0.0)] + [(f, *hips_xz(p)) for f, p in keys]
    meta = {"op": "vault", "height_m": h, "approach_m": a, "speed": speed, "takeoff_frame": f_plant,
            "land_frame": f_land, "obstacle": {"near_z": a, "depth": depth, "height": h}}
    return solve(out, f_end + 1, keys, (), rp, seed, 100, meta, plant_hands)


LIMP = {"right": {"turns": {"Hips": [["z", -5]], "Spine1": [["z", -4]], "Spine2": [["z", -4], ["x", 4]],
                            "Chest": [["z", -4], ["x", 6]], "Head": [["z", 7]],
                            "RightLeg": [["x", 8], ["z", 4]], "RightFoot": [["x", -8]],
                            "LeftLeg": [["x", -6]], "LeftShin": [["x", 12]], "LeftFoot": [["x", -6]],
                            "RightArm": [["z", -8], ["x", -6]], "LeftArm": [["z", 12]],
                            "RightForeArm": [["x", -20]], "LeftForeArm": [["x", -10]]}}}
LIMP["left"] = {"base": None, **LIMP["right"], "mirror": True}


def locomotion(out, path, speed=1.5, style="none", style_pose=None, seed=0):
    """along a polyline of [forward, left] points (m) from the origin at `speed` m/s: an eased start and stop on the
    root path, standing keys at both ends; a style pose (a limp: ours, from LIMP) laid down every stride, alternating
    with nothing, so Kimodo walks through it"""
    pts = np.array([[0.0, 0.0]] + [[float(q[1]), float(q[0])] for q in path])   # (x = left, z = forward)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if seg.sum() < 0.05:
        raise ValueError("the path is too short")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    lead, tail = 10, 18
    move = int(math.ceil(cum[-1] / float(speed) * 30 * 1.15))            # 15 % for easing in and out
    n = min(300, lead + move + tail)
    move = n - lead - tail

    def at_s(s):
        return np.array([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])])

    rp = []
    for f in range(n):
        u = np.clip((f - lead) / move, 0, 1)
        x, z = at_s(cum[-1] * u * u * (3 - 2 * u))
        rp.append((f, x, z))
    d0, d1 = pts[1] - pts[0], pts[-1] - pts[-2]
    yaw0, yaw1 = math.degrees(math.atan2(d0[0], d0[1])), math.degrees(math.atan2(d1[0], d1[1]))
    keys = [(0, at(S.build("stand"), yaw=yaw0)), (n - 1, at(S.build("stand"), *pts[-1], yaw=yaw1))]
    spec = style_pose if style_pose is not None else LIMP.get(style.replace("limp_", "")) if style != "none" else None
    if spec is not None:
        stride = int(round(30 * (1.1 if style != "none" else 1.0)))          # a slow, uneven gait: ~1.1 s a stride
        for f in range(lead + stride // 2, n - tail - 6, stride):
            s0 = np.array(rp[max(f - 2, 0)][1:])
            s1 = np.array(rp[min(f + 2, n - 1)][1:])
            d = s1 - s0
            yaw = math.degrees(math.atan2(d[0], d[1])) if np.linalg.norm(d) > 1e-4 else yaw1
            keys.append((f, at(S.build(spec), *rp[f][1:], yaw=yaw)))
    meta = {"op": "locomotion", "length_m": round(float(cum[-1]), 3), "speed_ms": float(speed), "style": style}
    return solve(out, n, keys, (), rp, seed, 100, meta)


proto.serve({"generate": generate, "vault": vault, "locomotion": locomotion},
            {"worker": "kimodo", "model": MODEL, "load_s": LOAD_S, "skeleton": model.skeleton.nbjoints})
