"""Key-pose recipes, pure numpy: the same keys feed Kimodo (kimodo_worker.py) and the CPU blocking preview
(blocking.py), so the poses can be checked on the freak before any GPU time is spent. Model space as in soma.py:
x = the body's left, y = up, z = forward, meters on SOMA. Every pose is authored here from the T-pose (our own).

A recipe returns {"n", "keys": [(frame, pose)], "effectors", "root_path": [(frame, x, z)], "meta", "hands"};
"hands" are plants [{"joint", "in", "from", "to", "out", "pos"}]: eased on from `in`, held `from`..`to`, eased off by `out`. They are
laid on after generation (Kimodo lets a palm drift with the body) and again on the freak by blender/retarget.py (its
arms follow rotations, not positions)."""
import json
import math

import numpy as np

import soma as S


def at(pose, x=0.0, z=0.0, yaw=0.0, hips_y=None):
    """a pose stood at (x, z) facing yaw (deg left of +Z), its hips raised to hips_y if given (an airborne key)"""
    p = S.place(pose, yaw=yaw)
    p[0, 0] += x
    p[0, 2] += z
    if hips_y is not None:
        p[0, 1] += hips_y - S.fk(p)[0, 1]
    return p


def sole(p):
    """the lowest point of the feet (m): toe tips, and the heels 6.8 cm under the ankles"""
    W = S.fk(p)
    return min(min(W[S.JI[s + "ToeEnd"], 1], W[S.JI[s + "Foot"], 1] - 0.068) for s in ("Left", "Right"))


def grounded(p):
    """a key with a foot down: the body moved so its lowest sole point is on the floor"""
    p = p.copy()
    p[0, 1] += 0.004 - sole(p)
    return p


def hips_xz(p):
    return S.fk(p)[0, [0, 2]]


def plant(poses, hands):
    """lay the plants on (model-space poses): each {"joint": LeftHand | RightHand | LeftFoot | RightFoot, "in", "from",
    "to", "out", "pos"} eased on with smoothstep weights, held, eased off, so no snap; the limb re-solved around its own
    middle joint, the end keeping its turn. Out of reach, the limb reaches for it: a push-off"""
    out = list(poses)
    for h in hands:
        j = h["joint"]
        up, lo = S.CHAIN[j]
        for f in range(max(h["in"], 0), min(h["out"], len(out) - 1) + 1):
            if f < h["from"]:
                u = (f - h["in"]) / max(h["from"] - h["in"], 1)
            elif f <= h["to"]:
                u = 1.0
            else:
                u = 1 - (f - h["to"]) / max(h["out"] - h["to"], 1)
            w = u * u * (3 - 2 * u)
            p = out[f].copy()
            W = S.fk(p)
            cur = W[S.JI[j], :3]
            goal = cur + w * (np.asarray(h["pos"]) - cur)
            mid = (W[S.JI[up], :3] + cur) / 2
            S.two_bone(p, up, lo, j, goal, W[S.JI[lo], :3] - mid)
            out[f] = p
    return out


def keyed(n, keys, effectors=(), root_path="keys"):
    """the generic recipe (motion_from_keys): keys [{frame, pose}], effectors [{joint, frame, position, pose?}],
    root_path "keys" | "none" | [[frame, left, forward], ...]. An effector's body is its own pose or the nearest key's,
    with the joint laid on its target"""
    built = [(int(k["frame"]), S.build(k["pose"])) for k in keys]
    eff = []
    for e in effectors:
        f = int(e["frame"])
        if e.get("pose") is not None:
            p = S.build(e["pose"])
        else:
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
    hands = [{"joint": j, "in": f - 2, "from": f, "to": f, "out": f + 2, "pos": [float(x) for x in e["position"]]}
             for (j, f, _), e in zip(eff, effectors)]
    return {"n": int(n), "keys": built, "effectors": eff, "root_path": rp, "hands": hands,
            "meta": {"op": "generate", "hand_targets": [h for h in hands if "Hand" in h["joint"]]}}


# ---- the vault: a one-hand tuck vault ------------------------------------------------------------------------------
# Blocked like a hand-keyed vault: the take-off from the far foot with the near knee driven up; the plant (the bone
# hand, left, on the top outside the knees, the body low and pitched over it, the legs still trailing); the tuck (both
# knees pulled to the chest over the near edge as the body turns away from the hand and leans onto it); the cross (the
# hips over the top, the legs tucked over it, the hand letting go); the drop (the legs coming down past the far edge);
# a two-foot landing in a crouch; the rise. The revolver arm swings up and out for balance and never touches the box.
# SKULL's shoulders sit 1.7 studs higher than a SOMA body's over arms of the same reach: he only gets a hand on the top
# pitched well forward, and his wide shoulders want the plant outside his knees. His feet are tall (the ankle 0.89
# stud over the sole, a SOMA ankle 0.27): over the box the tuck is tight, the ankles at the hips' height.
VAULT_TAKEOFF = {"turns": {
    "LeftLeg": [["x", -20]], "LeftShin": [["x", 45]], "LeftFoot": [["x", -25]],          # the take-off foot under him
    "RightLeg": [["x", -80]], "RightShin": [["x", 140]], "RightFoot": [["x", -40]],      # the near knee driven up
    "Hips": [["x", 15]], "Spine2": [["x", 6]], "Chest": [["x", 8]], "Head": [["x", -18]],
    "LeftArm": [["x", -55], ["z", 10]], "LeftForeArm": [["x", -15]],                     # the bone hand reaching ahead
    "RightArm": [["x", 45], ["z", -15]], "RightForeArm": [["x", -45]]}}                  # the revolver swung back
VAULT_PLANT = {"turns": {
    "LeftLeg": [["x", 25]], "LeftShin": [["x", 35]], "LeftFoot": [["x", 20]],            # the take-off leg trailing
    "RightLeg": [["x", -75]], "RightShin": [["x", 125]], "RightFoot": [["x", -30]],      # the knee still driving
    "Hips": [["x", 28]], "Spine1": [["x", 10]], "Spine2": [["x", 12], ["z", -6]], "Chest": [["x", 12], ["z", -6]],
    "Head": [["x", -40]],
    "RightArm": [["z", -50], ["x", 20]], "RightForeArm": [["x", -50]]}}                  # the revolver swinging up
VAULT_TUCK = {"legs": [100, 75, 15], "turns": {                  # knees to the chest, heels to the seat
    "Hips": [["x", 15]], "Spine1": [["z", -10], ["x", 6]], "Spine2": [["z", -12], ["x", 8]], "Chest": [["z", -12], ["x", 8]],
    "Head": [["x", -30], ["z", 10]],
    "RightArm": [["z", -85], ["x", -10]], "RightForeArm": [["x", -55]]}}                 # the revolver up and out
VAULT_CROSS = {"legs": [100, 75, -5], "turns": {
    "Hips": [["x", -5]], "Spine1": [["z", -12]], "Spine2": [["z", -12]], "Chest": [["z", -10], ["x", -6]],
    "Head": [["x", -10], ["z", 12]],
    "LeftArm": [["z", 25], ["x", 20]], "LeftForeArm": [["x", -25]],                      # the hand off the top, back
    "RightArm": [["z", -105], ["x", -10]], "RightForeArm": [["x", -40]]}}
VAULT_DROP = {"legs": [70, 50, 5], "turns": {
    "Spine2": [["z", -6]], "Chest": [["z", -5], ["x", 6]], "Head": [["x", -12], ["z", 5]],
    "LeftArm": [["z", 35]], "LeftForeArm": [["x", -20]],
    "RightArm": [["z", -55]], "RightForeArm": [["x", -35]]}}


# ---- the hurdle: a running jump over the obstacle, no hands (the default vault) -------------------------------------
# What the owner liked in the probe (MotionBricks' jump over 0.5 m, 2026-10-27), blocked from our own poses: the
# take-off from the left foot with the right knee attacking; the lead (right) leg reaching over the top, the trail leg
# swung out to the side with its knee bent, the trunk pitched over the lead leg, the bone hand forward, the revolver
# back; the trail knee pulled through as the lead foot comes down; the landing on the lead foot; the run-out and stop.
# The feet are placed like IK controls (the knees solved toward poles): picked by angle, the lead foot swept down
# into the top on its way through. Flight is ballistic at the run's speed: ~13 frames over 1.55 m.
# The hurdle's keys: the trunk and arms here, the ankles placed in hurdle() by IK. Right leg leads, the left takes off;
# the arm opposite the lead leg (the bone hand) reaches along it, the revolver arm counters it.
HURDLE_OFF = {"turns": {                          # the take-off: the left leg driving long behind, its toes last off
    "Hips": [["x", 10]], "Spine2": [["x", 4]], "Chest": [["x", 4]], "Head": [["x", -14]],
    "LeftFoot": [["x", 30]], "RightFoot": [["x", -15]],
    "LeftArm": [["x", -60], ["z", 6]], "LeftForeArm": [["x", -50]],
    "RightArm": [["x", 45], ["z", -10]], "RightForeArm": [["x", -60]]}}
HURDLE_ATTACK = {"turns": {                       # the lead knee above the hip, its heel tucked under it
    "Hips": [["x", 12]], "Spine2": [["x", 6]], "Chest": [["x", 6]], "Head": [["x", -18]],
    "LeftFoot": [["x", 35]], "RightFoot": [["x", -10]],
    "LeftArm": [["x", -75]], "LeftForeArm": [["x", -25]],
    "RightArm": [["x", 50], ["z", -12]], "RightForeArm": [["x", -65]]}}
HURDLE_LEAD = {"turns": {                         # over the top: the lead leg long, the chest down toward its knee
    "Hips": [["x", 16]], "Spine1": [["x", 6]], "Spine2": [["x", 8]], "Chest": [["x", 8]], "Head": [["x", -30]],
    "LeftFoot": [["x", 20]], "RightFoot": [["x", -15]],
    "LeftArm": [["x", -85], ["z", -8]], "LeftForeArm": [["x", -10]],
    "RightArm": [["x", 35], ["z", -20]], "RightForeArm": [["x", -70]]}}
HURDLE_CLEAR = {"turns": {                        # the lead leg reaching down, the trail knee coming round the side
    "Hips": [["x", 10]], "Spine2": [["x", 7]], "Chest": [["x", 6]], "Head": [["x", -18]],
    "LeftFoot": [["x", -15]], "RightFoot": [["x", -10]],
    "LeftArm": [["x", 15], ["z", 35]], "LeftForeArm": [["x", -60]],                       # the bone hand swept back, wide
    "RightArm": [["x", -45], ["z", -10]], "RightForeArm": [["x", -70]]}}                 # the revolver forward
HURDLE_LAND = {"turns": {                         # down on the lead foot's ball, the trail knee driving on high
    "Hips": [["x", 10]], "Spine2": [["x", 5]], "Chest": [["x", 5]], "Head": [["x", -12]],
    "RightFoot": [["x", 15]], "LeftFoot": [["x", -10]],
    "LeftArm": [["x", 35], ["z", 10]], "LeftForeArm": [["x", -60]],
    "RightArm": [["x", -40], ["z", -10]], "RightForeArm": [["x", -70]]}}


def feet(p, right=None, left=None):
    """lay the ankles on (target, knee pole) like IK controls; each foot keeps its turn"""
    for side, t in (("Right", right), ("Left", left)):
        if t is not None:
            S.two_bone(p, side + "Leg", side + "Shin", side + "Foot", t[0], t[1])
    return p


RUN_TRUNK = {"Hips": [["x", 8]], "Spine2": [["x", 5]], "Chest": [["x", 6]], "Head": [["x", -10]]}


def run_step(side, z, reach=0.28, speed=1.0):
    """a run's contact pose at hips z: the `side` foot planting ahead (feet by IK), the other leg folded behind, the
    arms swinging opposite (the other arm forward); speed scales the stride for a jog or a slowing step"""
    o = "Left" if side == "Right" else "Right"
    sx = 1.0 if side == "Left" else -1.0             # +x is the body's left
    spec = {"turns": {**RUN_TRUNK, o + "Arm": [["x", -45 * speed]], o + "ForeArm": [["x", -60]],
                      side + "Arm": [["x", 40 * speed]], side + "ForeArm": [["x", -45]]}}
    p = at(S.build(spec), z=z, hips_y=0.95)
    feet(p, **{side.lower(): ((0.1 * sx, 0.09, z + reach * speed), (0, 0.2, 1)),
               o.lower(): ((-0.1 * sx, 0.09 + 0.3 * speed, z - 0.42 * speed), (0, -0.6, 1))})
    return grounded(p)


def run_mid(side, zf, z):
    """a run's mid-stance: the hips at z over the `side` foot planted flat at zf, the other knee passing under them"""
    o = "Left" if side == "Right" else "Right"
    sx = 1.0 if side == "Left" else -1.0
    spec = {"turns": {**RUN_TRUNK, side + "Arm": [["x", 10]], side + "ForeArm": [["x", -70]],
                      o + "Arm": [["x", -10]], o + "ForeArm": [["x", -70]]}}
    p = at(S.build(spec), z=z, hips_y=0.94)
    feet(p, **{side.lower(): ((0.1 * sx, 0.075, zf), (0, 0, 1)), o.lower(): ((-0.1 * sx, 0.40, z - 0.15), (0, -0.2, 1))})
    return grounded(p)


def run_push(side, zf, z):
    """a run's push-off: the hips at z, the `side` foot behind at zf with its toes last on the floor, the other knee
    driving forward, the arms swung the other way round from the contact"""
    o = "Left" if side == "Right" else "Right"
    sx = 1.0 if side == "Left" else -1.0
    spec = {"turns": {**RUN_TRUNK, side + "Foot": [["x", 30]], side + "Arm": [["x", -45]], side + "ForeArm": [["x", -60]],
                      o + "Arm": [["x", 40]], o + "ForeArm": [["x", -45]]}}
    p = at(S.build(spec), z=z, hips_y=0.97)
    feet(p, **{side.lower(): ((0.1 * sx, 0.13, zf), (0, 0, 1)), o.lower(): ((-0.1 * sx, 0.29, z + 0.11), (0, 0.2, 1))})
    return grounded(p)


def hurdle(height=0.64, speed="run", approach=2.55):
    v = 3.4 if speed == "run" else 2.2               # a walk can't leap: it is a slower, shorter run-up
    h, depth = float(height), 0.35
    step = v / 3                                     # a stride every 10 frames
    u = v / S.FPS                                    # m a frame
    n_run = max(1, int(round((float(approach) - 1.38) / step)))
    f_off = 10 * n_run + 6                           # take-off: the left toes leave the floor
    a = u * f_off + 0.7                              # the near face, 0.7 m past the hips at the take-off
    # the flight is ballistic (gravity only, nothing hovers): the hips must peak 0.58 m over the top for SKULL's lead
    # foot to clear it, which sets the time in the air; they cover take-off to landing (0.4 m past the far face) at
    # an even speed. At a run that is the run's own 3.4 m/s
    z0, dist = a - 0.7, 0.7 + depth + 0.4
    T = math.sqrt(8 * (max(1.22, h + 0.58) - 1.0) / 9.81)
    n = max(8, round(T * S.FPS))
    T = n / S.FPS
    f_attack, f_lead, f_clear, f_land = (f_off + round(n * k) for k in (0.15, 0.38, 0.77, 1.0))
    f_step, f_end = f_land + 9, f_land + 22
    # the run-in, blocked as a real run: each stride's contact, mid-stance and push-off with the foot planted on one
    # spot throughout and 0.13 s in the air between strides, ending on the left foot's take-off stride
    run = []
    for k in range(n_run + 1):
        side, fc = "Left" if (n_run - k) % 2 == 0 else "Right", 10 * k
        zf = u * fc + 0.28                           # where this foot lands (run_step's reach)
        run += [(fc, run_step(side, u * fc)), (fc + 3, run_mid(side, zf, u * (fc + 3)))]
        if k < n_run:
            run.append((fc + 6, run_push(side, zf, u * (fc + 6))))
    # the ankles are placed like IK controls (model space), the knees solved toward their poles
    off = grounded(feet(at(S.build(HURDLE_OFF), z=z0, hips_y=1.0),
                        left=((0.1, 0.13, z0 - 0.4), (0, 0, 1)), right=((-0.1, 0.46, z0 + 0.18), (0, 0.3, 1))))
    z1 = z0 + dist
    land = grounded(feet(at(S.build(HURDLE_LAND), z=z1, hips_y=1.0),
                         right=((-0.1, 0.1, z1 + 0.1), (0, 0.2, 1)), left=((0.15, 0.55, z1 + 0.1), (0.3, 0, 1))))
    y0, y1 = S.fk(off)[0, 1], S.fk(land)[0, 1]

    def air(f, spec, right, left):
        """an airborne key: the hips on the arc, each ankle at (x, up, forward) from the hips with its knee's pole"""
        t = (f - f_off) / S.FPS
        y, z = y0 + (y1 - y0) * t / T + 0.5 * 9.81 * t * (T - t), z0 + dist * t / T
        rel = lambda r: ((r[0][0], y + r[0][1], z + r[0][2]), r[1])
        return feet(at(S.build(spec), z=z, hips_y=y), right=rel(right), left=rel(left))

    attack = air(f_attack, HURDLE_ATTACK, ((-0.1, -0.26, 0.22), (0, 1, 0.3)),         # heel tucked under the knee
                 ((0.12, -0.72, -0.45), (0, -0.3, 1)))                                # the take-off leg trailing
    lead = air(f_lead, HURDLE_LEAD, ((-0.08, -0.30, 0.80), (0, 1, 0)),                # the lead leg long over the top
               ((0.33, -0.38, -0.47), (1, -0.6, 0)))                                  # the trail knee folding out
    clear = air(f_clear, HURDLE_CLEAR, ((-0.1, -0.84, 0.39), (0, 0.3, 1)),            # the lead foot reaching down
                ((0.35, -0.24, -0.26), (1, 0.1, 0.9)))                                # the trail knee out and forward
    # the run-out: one slowing step on the left, then the stop
    out_step = run_step("Left", z1 + 0.75 * step / 1.13, speed=0.7)
    end = grounded(at(S.build("ready"), z=z1 + 1.15 * step / 1.13))
    keys = run + [(f_off, off), (f_attack, attack), (f_lead, lead), (f_clear, clear), (f_land, land),
                  (f_step, out_step), (f_end, end)]
    meta = {"op": "vault", "style": "hurdle", "height_m": h, "approach_m": round(a, 3), "speed": speed,
            "takeoff_frame": f_off, "land_frame": f_land, "obstacle": {"near_z": a, "depth": depth, "height": h}}
    return {"n": f_end + 1, "keys": keys, "effectors": [], "hands": [], "meta": meta,
            "root_path": [(f, *hips_xz(p)) for f, p in keys]}


def edited(file):
    """a take's keys.json with its poses changed in the pose editor; the hips' ground path follows the keys"""
    with open(file) as fh:
        d = json.load(fh)
    keys = sorted(((int(k["frame"]), np.asarray(k["pose"], float)) for k in d["keys"]), key=lambda k: k[0])
    eff = [(e["joint"], int(e["frame"]), np.asarray(e["pose"], float)) for e in d["effectors"]]
    return {"n": int(d["n"]), "keys": keys, "effectors": eff, "hands": d["hands"], "meta": d["meta"],
            "root_path": [(f, *hips_xz(p)) for f, p in keys] if len(keys) > 1 else None}


def vault(height=0.64, speed="run", approach=2.55, hand="left", style="hurdle"):
    """over an obstacle `height` m tall, 0.35 m deep, whose near face is `approach` m ahead of the start: style hurdle (a
    running jump, no hands: the default) or hand (a one-hand tuck vault, for taller obstacles)"""
    return hurdle(height, speed, approach) if style == "hurdle" else hand_vault(height, speed, approach, hand)


def hand_vault(height=0.64, speed="run", approach=2.55, hand="left"):
    """over an obstacle `height` m tall, 0.35 m deep, whose near face is `approach` m ahead of the start"""
    run = speed == "run"
    v = 3.4 if run else 1.6                          # m/s on SOMA (SKULL's run is 13 studs/s)
    k = 1.0 if run else 1.5                          # the walk's vault, slower throughout
    h, a, depth = float(height), float(approach), 0.35
    side = "Left" if hand == "left" else "Right"
    sx = 1.0 if hand == "left" else -1.0             # the planted hand's side (+x is the body's left)

    def mirror(spec):                                # the recipe is written for the left hand
        return spec if hand == "left" else {"base": None, **spec, "mirror": True}

    def t(n):
        return int(round(n * k))

    f_off = max(10, int(round((a - 0.85) / v * 30)))  # take-off: the hips 0.85 m before the near face
    f_plant = f_off + t(4)
    f_tuck = f_plant + t(3)
    f_cross = f_tuck + t(4)
    f_drop = f_cross + t(4)
    f_land = f_drop + t(5)
    f_end = f_land + t(24)
    hand_at = np.array([0.28 * sx, h + 0.03, a + 0.12])  # the palm on the top past the near edge, outside the knees

    off = grounded(at(S.build(mirror(VAULT_TAKEOFF)), z=a - 0.85))
    plant = at(S.build(mirror(VAULT_PLANT)), z=a - 0.6, yaw=-8 * sx, hips_y=h + 0.24)
    S.two_bone(plant, *S.CHAIN[side + "Hand"], side + "Hand", hand_at, (0.5 * sx, 0.3, -1))
    tuck = at(S.build(mirror(VAULT_TUCK)), x=-0.04 * sx, z=a - 0.16, yaw=-22 * sx, hips_y=h + 0.42)
    S.two_bone(tuck, *S.CHAIN[side + "Hand"], side + "Hand", hand_at, (0.6 * sx, 0.3, -0.8))
    cross = at(S.build(mirror(VAULT_CROSS)), x=-0.08 * sx, z=a + depth * 0.6, yaw=-35 * sx, hips_y=h + 0.45)
    drop = at(S.build(mirror(VAULT_DROP)), x=-0.06 * sx, z=a + depth + 0.42, yaw=-15 * sx, hips_y=h + 0.22)
    land = at(S.build("crouch_land"), z=a + depth + 0.75, yaw=-5 * sx)
    end = at(S.build("ready"), z=a + depth + 0.75 + (1.6 if run else 0.8))
    keys = [(f_off, off), (f_plant, plant), (f_tuck, tuck), (f_cross, cross), (f_drop, drop), (f_land, land),
            (f_end, end)]
    # the palm on the top from the plant through the tuck, let go as the hips pass over it
    hands = [{"joint": side + "Hand", "in": f_plant - 3, "from": f_plant, "to": f_tuck, "out": f_cross,
              "pos": hand_at.round(4).tolist()}]
    meta = {"op": "vault", "style": "hand", "height_m": h, "approach_m": a, "speed": speed, "hand": hand, "takeoff_frame": f_off,
            "land_frame": f_land, "obstacle": {"near_z": a, "depth": depth, "height": h}, "hand_targets": hands}
    return {"n": f_end + 1, "keys": keys, "effectors": [], "hands": hands, "meta": meta,
            "root_path": [(0, 0.0, 0.0)] + [(f, *hips_xz(p)) for f, p in keys]}


if __name__ == "__main__":                          # python recipes.py <skeleton.npz>: the vault's keys checked
    import sys
    S.load_skeleton(sys.argv[1])
    for hand, style in (("left", "hurdle"), ("left", "hand"), ("right", "hand")):
        r = vault(hand=hand, style=style)
        ob = r["meta"]["obstacle"]
        h, a, dep = ob["height"], ob["near_z"], ob["depth"]
        for f, p in r["keys"]:
            W = S.fk(p)
            low = sole(p)
            legs = [W[S.JI[s + n], :3] for s in ("Left", "Right") for n in ("Shin", "Foot", "ToeEnd")]
            over = [y for x, y, z in legs if a - 0.05 < z < a + dep + 0.05]
            sh = W[S.JI[("Left" if hand == "left" else "Right") + "Arm"], :3]
            palm = r["hands"][0]["pos"] if r["hands"] else sh
            print(f"{style}/{hand} f{f:3d} hips {W[0, :3].round(2)} sole {low:+.2f} heading {S.heading(p):6.1f} "
                  f"legs over box min y {min(over) if over else None} shoulder->palm "
                  f"{np.linalg.norm(sh - np.array(palm)):.2f}")
            assert low > -0.02, (f, low)                 # no key sinks into the floor
            assert not over or min(over) > h + 0.05, (f, min(over))
