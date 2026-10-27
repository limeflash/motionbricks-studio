"""SOMA posing shared by both workers (numpy + scipy only), ported from the 2026-10-27 probe
(D:\\Pictures\\build a freak\\forms\\skull_anim\\motionbricks\\scripts\\mb_generate.py and mb_moves.py).

Model space: Y up, the body faces +Z, its left is +X, meters (SOMA: leg 0.857 m). A pose is [77, 7] local transforms
(px, py, pz, qx, qy, qz, qw) on MotionBricks' 77-joint SOMA skeleton; only the root's position is used.

Pose specs (what the calling Claude writes) are JSON, always built from the straight-legged STAND:
  {"base": "ready",                        a library pose (POSES) whose spec this one extends
   "legs": [thigh, shin, root_pitch],      knees bent (deg), feet kept flat, the hips dropped so the feet stay down
   "turns": {"Chest": [["x", 12]]},        extra world-axis turns per joint, in order (children follow):
                                           x+ pitches a trunk forward / swings a hanging limb back, y+ turns left,
                                           z+ rolls the left side up
   "shift": [left, up, forward],           root offset (m)
   "ik": [{"joint": "LeftHand", "target": [x, y, z], "pole": [x, y, z]}],   two-bone IK in model space
   "mirror": false,                        left for right
   "root": {"forward": 0, "left": 0, "up": 0, "yaw": 0}}   where the body has travelled to (m, deg left)

BONES sample poses are never used: every library pose here is authored from the skeleton's own T-pose.
"""
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

FPS = 30
DT = 1.0 / FPS
LEG = 0.857                                         # SOMA thigh 0.433 + shin 0.424
STUD = LEG / 3.358                                  # one SKULL stud on SOMA, by leg length (heavy legs as drawn)

NAMES, PAR, REF, M, MI, JI = [], [], None, None, None, {}


def load_skeleton(path):
    """the skeleton dumped by mb_worker (names, parents, reference locals, model -> runtime transform)"""
    global NAMES, PAR, REF, M, MI, JI, STAND
    z = np.load(path)
    NAMES[:] = [str(n) for n in z["names"]]
    PAR[:] = [int(p) for p in z["parents"]]
    REF = z["ref"].astype(np.float64)
    M = z["m"].astype(np.float64)
    MI = tinv(M)
    JI.clear()
    JI.update({n: i for i, n in enumerate(NAMES)})
    STAND = posed(REF, STAND_SPEC)


# ---- transforms ------------------------------------------------------------------------------------------------------
def tmul(a, b):
    ra = R.from_quat(a[3:])
    return np.concatenate([a[:3] + ra.apply(b[:3]), (ra * R.from_quat(b[3:])).as_quat()])


def tinv(a):
    ri = R.from_quat(a[3:]).inv()
    return np.concatenate([-ri.apply(a[:3]), ri.as_quat()])


def fk(pose):
    out = np.zeros((len(NAMES), 7))
    for j in range(len(NAMES)):
        out[j] = pose[j] if PAR[j] < 0 else tmul(out[PAR[j]], pose[j])
    return out


def to_world(pose):
    p = pose.copy()
    p[0] = tmul(M, pose[0])
    return p


def from_world(pose):
    p = pose.copy()
    p[0] = tmul(MI, pose[0])
    return p


# ---- posing ----------------------------------------------------------------------------------------------------------
def posed(base, spec, shift=(0, 0, 0)):
    """base locals with extra world-axis turns per joint (spec: name -> [(axis, deg), ...]); children follow"""
    out = base.copy()
    W = [None] * len(NAMES)
    for j, n in enumerate(NAMES):
        Wp = W[PAR[j]] if PAR[j] >= 0 else R.identity()
        Wj = Wp * R.from_quat(base[j, 3:])
        for ax, deg in spec.get(n, ()):
            Wj = R.from_euler(ax, deg, degrees=True) * Wj
        out[j, 3:] = (Wp.inv() * Wj).as_quat()
        W[j] = Wj
    out[0, :3] += shift
    return out


def legs(thigh, shin, root_pitch=0.0):
    """knees bent: thigh forward / shin back (deg, world X turns after the root's pitch), the foot kept flat"""
    d = {}
    for s in ("Left", "Right"):
        d[s + "Leg"] = [("x", -thigh - root_pitch)]
        d[s + "Shin"] = [("x", thigh + shin)]
        d[s + "Foot"] = [("x", -shin)]
    return d


def drop_for(thigh, shin):
    a, b = math.radians(thigh), math.radians(shin)
    return 0.433 * (1 - math.cos(a)) + 0.424 * (1 - math.cos(b))


def blend(a, b, w, back=0.0):
    out = a.copy()
    for j in range(len(NAMES)):
        out[j, 3:] = R.from_quat([a[j, 3:], b[j, 3:]]).mean(weights=[1 - w, w]).as_quat()
    out[0, :3] = a[0, :3] * (1 - w) + b[0, :3] * w
    out[0, 2] -= back
    return out


def place(pose, forward=0.0, left=0.0, up=0.0, yaw=0.0):
    """move only the root: forward / left / up meters, turned `yaw` degrees left about the hips"""
    p = pose.copy()
    ry = R.from_euler("y", yaw, degrees=True)
    p[0, 3:] = (ry * R.from_quat(p[0, 3:])).as_quat()
    p[0, :3] += (left, up, forward)
    return p


def set_world_rot(pose, j, rot):
    w = fk(pose)
    wp_ = R.from_quat(w[PAR[j], 3:]) if PAR[j] >= 0 else R.identity()
    pose[j, 3:] = (wp_.inv() * rot).as_quat()


def aim(pose, j, child, target):
    """turn joint j by the shortest rotation so its child lands on the line to target"""
    w = fk(pose)
    d0 = w[JI[child], :3] - w[JI[j], :3]
    d1 = np.asarray(target) - w[JI[j], :3]
    v = np.cross(d0, d1)
    s, c = np.linalg.norm(v), np.dot(d0, d1)
    if s < 1e-9:
        return
    set_world_rot(pose, JI[j], R.from_rotvec(v / s * np.arctan2(s, c)) * R.from_quat(w[JI[j], 3:]))


CHAIN = {"LeftHand": ("LeftArm", "LeftForeArm"), "RightHand": ("RightArm", "RightForeArm"),
         "LeftFoot": ("LeftLeg", "LeftShin"), "RightFoot": ("RightLeg", "RightShin")}


def two_bone(pose, upper, lower, end, target, pole):
    """analytic two-bone IK: the end on target (clamped to reach), the middle joint toward `pole` (a direction); the
    end keeps its world turn"""
    keep = R.from_quat(fk(pose)[JI[end], 3:])
    w = fk(pose)
    a, b, e = w[JI[upper], :3], w[JI[lower], :3], w[JI[end], :3]
    l1, l2 = np.linalg.norm(b - a), np.linalg.norm(e - b)
    t = np.asarray(target, float)
    d = t - a
    dist = np.clip(np.linalg.norm(d), abs(l1 - l2) + 1e-4, (l1 + l2) * 0.999)
    d = d / np.linalg.norm(d)
    t = a + d * dist
    x = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = np.sqrt(max(l1 * l1 - x * x, 0.0))
    p = np.asarray(pole, float) - np.dot(pole, d) * d
    p = p / max(np.linalg.norm(p), 1e-9)
    aim(pose, upper, lower, a + d * x + p * h)
    aim(pose, lower, end, t)
    set_world_rot(pose, JI[end], keep)


def counterpart(n):
    return "Right" + n[4:] if n.startswith("Left") else "Left" + n[5:] if n.startswith("Right") else n


MIRROR = np.diag([-1.0, 1.0, 1.0])


def mirror_pose(p):
    """left for right: each joint takes its counterpart's world turn from the (symmetric) T-pose, reflected in the
    body's mid plane; the hips move to -x"""
    w, wref = fk(p), fk(REF)
    W = []
    for j, n in enumerate(NAMES):
        c = JI[counterpart(n)]
        d = R.from_quat(w[c, 3:]) * R.from_quat(wref[c, 3:]).inv()
        W.append(R.from_matrix(MIRROR @ d.as_matrix() @ MIRROR) * R.from_quat(wref[j, 3:]))
    out = REF.copy()
    for j in range(len(NAMES)):
        out[j, 3:] = ((W[PAR[j]].inv() if PAR[j] >= 0 else R.identity()) * W[j]).as_quat()
    out[0, :3] = p[0, :3] * np.array([-1.0, 1.0, 1.0])
    return out


def heading(model_pose):
    """yaw (deg, from +Z toward +X) the pose faces: the hips' left-right line"""
    W = fk(model_pose)
    left = W[JI["LeftLeg"], :3] - W[JI["RightLeg"], :3]
    f = np.cross(left, [0, 1.0, 0])
    return math.degrees(math.atan2(f[0], f[2]))


# ---- the pose library: authored here from the T-pose (the probe's READY / HIT / CROUCH, plus the hit variants) -------
GUARD_ARMS = {"LeftArm": [["x", -20], ["z", 6]], "RightArm": [["x", -20], ["z", -6]],
              "LeftForeArm": [["x", -75]], "RightForeArm": [["x", -75]]}


def _arms(extra):
    return {k: GUARD_ARMS[k] + extra.get(k, []) for k in GUARD_ARMS}


STAND_SPEC = {"LeftArm": [("z", -78)], "RightArm": [("z", 78)], "LeftForeArm": [("x", -15)],
              "RightForeArm": [("x", -15)]}
STAND = None
POSES = {
    "stand": ({}, "standing straight, arms down"),
    "ready": ({"legs": [10, 8], "turns": GUARD_ARMS}, "knees soft, forearms up in a low guard (the probe's READY)"),
    "hit_front": ({"legs": [16, 14, -8], "turns": {
        "Hips": [["x", -8]], "Spine1": [["x", -8]], "Spine2": [["x", -10]], "Chest": [["x", -12]], "Neck1": [["x", -10]],
        "Head": [["x", -22]], **_arms({"LeftArm": [["x", -8], ["z", 22]], "RightArm": [["x", -8], ["z", -22]],
                                       "LeftForeArm": [["x", 35]], "RightForeArm": [["x", 35]]})}},
        "struck from the front: pelvis tipped back, chest and head thrown back, arms flung forward and out, knees "
        "giving (the probe's HIT, owner-approved in the knockback)"),
    "crouch_land": ({"legs": [48, 52, 15], "turns": {
        "Hips": [["x", 15]], "Chest": [["x", 12]], "Head": [["x", -18]],
        **_arms({"LeftArm": [["z", 40]], "RightArm": [["z", -40]], "LeftForeArm": [["x", 25]],
                 "RightForeArm": [["x", 25]]})}},
        "landing a knock-back on its feet: deep crouch, chest over the knees, arms out for balance (the probe's CROUCH)"),
    "hit_back": ({"legs": [14, 12, 6], "turns": {
        "Hips": [["x", 6]], "Spine1": [["x", -6]], "Spine2": [["x", -8]], "Chest": [["x", -10]], "Neck1": [["x", -8]],
        "Head": [["x", -20]], **_arms({"LeftArm": [["x", 30], ["z", 25]], "RightArm": [["x", 30], ["z", -25]],
                                       "LeftForeArm": [["x", 40]], "RightForeArm": [["x", 40]]})}},
        "struck from behind: hips driven forward, the back arched, head whipped back, arms flung back and out"),
    "hit_left": ({"legs": [14, 12], "turns": {
        "Hips": [["z", 6]], "Spine1": [["z", 6]], "Spine2": [["z", 8], ["y", -6]], "Chest": [["z", 10], ["y", -8]],
        "Neck1": [["z", -6]], "Head": [["z", -14]],
        **_arms({"LeftArm": [["z", 35]], "RightArm": [["z", -30]], "LeftForeArm": [["x", 30]],
                 "RightForeArm": [["x", 25]]})}},
        "struck on the left side: the trunk bent and turned to the right, the head lagging left, arms out for balance"),
    "hit_right": ({"base": "hit_left", "mirror": True}, "struck on the right side (hit_left mirrored)"),
}


def build(spec):
    """a pose spec (dict, or a library name) -> pose [77, 7] in model space"""
    if isinstance(spec, str):
        spec = {"base": spec}
    spec = dict(spec)
    chain, base = [], spec.get("base")
    while base:
        if base not in POSES:
            raise ValueError(f"unknown base pose {base!r}; library: {', '.join(POSES)}")
        lib = dict(POSES[base][0])
        chain.append(lib)
        base = lib.get("base")
    merged = {"turns": {}, "shift": [0.0, 0.0, 0.0], "ik": [], "legs": None, "mirror": False, "root": {}}
    for s in list(reversed(chain)) + [spec]:      # the deepest base first, the caller's own spec last
        for j, t in (s.get("turns") or {}).items():
            merged["turns"].setdefault(j, []).extend(t)
        merged["shift"] = [a + b for a, b in zip(merged["shift"], s.get("shift") or (0, 0, 0))]
        merged["ik"] += s.get("ik") or []
        if s.get("legs") is not None:
            merged["legs"] = s["legs"]
        if s.get("mirror"):                         # each mirror in the chain flips once, applied last
            merged["mirror"] = not merged["mirror"]
        merged["root"].update(s.get("root") or {})
    for j in merged["turns"]:
        if j not in JI:
            raise ValueError(f"unknown joint {j!r}")
    turns = {j: [tuple(t) for t in v] for j, v in merged["turns"].items()}
    shift = np.array(merged["shift"], float)
    if merged["legs"]:
        lg = list(merged["legs"]) + [0.0] * (3 - len(merged["legs"]))
        for j, t in legs(*lg).items():
            turns[j] = t + turns.get(j, [])
        shift[1] -= drop_for(lg[0], lg[1])
    p = posed(STAND, turns, shift=shift)
    for k in merged["ik"]:
        up, lo = CHAIN[k["joint"]]
        pole = k.get("pole") or ((0, -1, 0.3) if "Hand" in k["joint"] else (0, 0, 1))
        two_bone(p, up, lo, k["joint"], k["target"], pole)
    if merged["mirror"]:
        p = mirror_pose(p)
    r = merged["root"]
    return place(p, r.get("forward", 0.0), r.get("left", 0.0), r.get("up", 0.0), r.get("yaw", 0.0))


# ---- takes: poses in model space, written as CMU-named BVH for the retarget ------------------------------------------
U = 13.957 / 0.857                                  # SOMA meters -> CMU units (the SOMA leg as long as the CMU actor's)
CMU = {"Hips": "Hips", "Spine1": "LowerBack", "Spine2": "Spine", "Chest": "Spine1", "Neck1": "Neck", "Neck2": "Neck1",
       "Head": "Head"}
for _s in ("Left", "Right"):
    CMU.update({_s + "Shoulder": _s + "Shoulder", _s + "Arm": _s + "Arm", _s + "ForeArm": _s + "ForeArm",
                _s + "Hand": _s + "Hand", _s + "HandMiddle1": _s + "HandIndex1", _s + "Leg": _s + "UpLeg",
                _s + "Shin": _s + "Leg", _s + "Foot": _s + "Foot", _s + "ToeBase": _s + "ToeBase"})
END = {"Head": "HeadEnd", "LeftHandMiddle1": "LeftHandMiddle2", "RightHandMiddle1": "RightHandMiddle2",
       "LeftToeBase": "LeftToeEnd", "RightToeBase": "RightToeEnd"}


def to_model(world_poses):
    """runtime (Z up) roots back to model space, the take turned to face +Z at its first frame, hips over 0"""
    poses = [from_world(p) for p in world_poses]
    a = R.from_euler("y", -heading(poses[0]), degrees=True)
    o = poses[0][0, :3].copy()
    for q in poses:
        q[0, :3] = a.apply(q[0, :3] - o * (1, 0, 1))
        q[0, 3:] = (a * R.from_quat(q[0, 3:])).as_quat()
    return poses


def write_bvh(path, poses):
    keep = [j for j, n in enumerate(NAMES) if n in CMU]
    kids = {j: [k for k in keep if PAR[k] == j] for j in keep}
    lines = ["HIERARCHY"]

    def emit(j, depth):
        ind = "\t" * depth
        off = REF[j, :3] * U if PAR[j] >= 0 else np.zeros(3)
        lines.append(f"{ind}{'ROOT' if PAR[j] < 0 else 'JOINT'} {CMU[NAMES[j]]}")
        lines.append(ind + "{")
        lines.append(f"{ind}\tOFFSET {off[0]:.5f} {off[1]:.5f} {off[2]:.5f}")
        ch = "6 Xposition Yposition Zposition Zrotation Yrotation Xrotation" if PAR[j] < 0 else \
            "3 Zrotation Yrotation Xrotation"
        lines.append(f"{ind}\tCHANNELS {ch}")
        for k in kids[j]:
            emit(k, depth + 1)
        if NAMES[j] in END:
            e = REF[JI[END[NAMES[j]]], :3] * U
            lines.extend([f"{ind}\tEnd Site", ind + "\t{", f"{ind}\t\tOFFSET {e[0]:.5f} {e[1]:.5f} {e[2]:.5f}",
                          ind + "\t}"])
        lines.append(ind + "}")

    emit(0, 0)
    frames = [REF] + list(poses)                    # frame 0: the T-pose, as in every cgspeed CMU file
    lines += ["MOTION", f"Frames: {len(frames)}", f"Frame Time: {DT:.7f}"]
    for p in frames:
        row = list(p[0, :3] * U)
        for j in keep:
            row += list(R.from_quat(p[j, 3:]).as_euler("ZYX", degrees=True))
        lines.append(" ".join(f"{v:.5f}" for v in row))
    with open(path, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")


# hinges as a body has them: the joint, the bone it turns on (whose frame carries the hinge axis, as in the T-pose:
# knees flex about the body's left, elbows forward) and how far it bends at most (deg)
HINGES = {"left knee": ("LeftLeg", "LeftShin", "LeftFoot", (1, 0, 0), 155),
          "right knee": ("RightLeg", "RightShin", "RightFoot", (1, 0, 0), 155),
          "left elbow": ("LeftArm", "LeftForeArm", "LeftHand", (0, -1, 0), 155),
          "right elbow": ("RightArm", "RightForeArm", "RightHand", (0, 1, 0), 155)}


def anatomy(W, ref):
    """what one frame (fk rows) asks of the joints that a body can't do: a knee or elbow bent too far, backwards or
    sideways off its hinge; a hip flexed past 145°, stretched back past 35°, spread past 75° or crossed past 30°
    (measured on Kimodo's own leaps, which reach 142° and 70°). An elbow's hinge is only read once it is well bent:
    the forearm's twist and carrying angle tilt it a little"""
    out = []
    for name, (up, mid, end, axis, most) in HINGES.items():
        t, s = W[JI[mid], :3] - W[JI[up], :3], W[JI[end], :3] - W[JI[mid], :3]
        bend = np.degrees(np.arccos(np.clip(np.dot(t, s) / np.linalg.norm(t) / np.linalg.norm(s), -1, 1)))
        if bend > most:
            out.append((name, f"bent {bend:.0f}° (a body stops at {most}°)"))
        elif bend > (25 if "knee" in name else 45):
            hinge = (R.from_quat(W[JI[up], 3:]) * R.from_quat(ref[JI[up], 3:]).inv()).apply(axis)
            off = np.degrees(np.arccos(np.clip(np.dot(np.cross(t, s) / np.linalg.norm(np.cross(t, s)), hinge), -1, 1)))
            if off > 50:
                out.append((name, "bends backwards" if off > 130 else f"bends sideways ({off:.0f}° off its hinge)"))
    pelvis = R.from_quat(W[0, 3:]) * R.from_quat(ref[0, 3:]).inv()
    for side, sx in (("left", 1), ("right", -1)):
        n = side.capitalize()
        d = pelvis.inv().apply(W[JI[n + "Shin"], :3] - W[JI[n + "Leg"], :3])
        d /= np.linalg.norm(d)
        flex, spread = np.degrees(np.arctan2(d[2], -d[1])), np.degrees(np.arcsin(np.clip(d[0] * sx, -1, 1)))
        for bad, what in ((flex > 145, f"flexed {flex:.0f}°"), (flex < -35, f"stretched back {-flex:.0f}°"),
                          (spread > 75, f"spread {spread:.0f}°"), (spread < -30, f"crossed {-spread:.0f}°")):
            if bad:
                out.append((side + " hip", what))
    return out


def flights(W, soles, held=()):
    """each stretch with both feet off the floor and no hand planted (`held` frames) is ballistic: from the last foot
    off to the first foot down (T s) the hips rise g T^2 / 8 over the line between them, give or take the limbs' swing.
    Higher hovers, lower floats down"""
    soles = [0.0 if f in held else y for f, y in enumerate(soles)]
    out, f = [], 0
    while f < len(W):
        if soles[f] <= 0.05:
            f += 1
            continue
        e = f
        while e < len(W) and soles[e] > 0.05:
            e += 1
        if f > 0 and e < len(W) and e - f >= 4:
            T = (e - f + 1) / FPS
            y = np.array([W[k][0, 1] for k in range(f - 1, e + 1)])
            rise = (y - np.linspace(y[0], y[-1], len(y))).max()
            want = 9.81 * T * T / 8
            if abs(rise - want) > max(0.06, 0.35 * want):
                out.append(f"frames {f}-{e - 1} are not a real flight: in {T:.2f} s in the air the hips rise "
                           f"{100 * rise:.0f} cm where gravity gives {100 * want:.0f} cm (the body "
                           f"{'hovers' if rise > want else 'floats down'}; re-time the keys or change the jump's height)")
        f = e
    return out


def checks(poses, meta):
    """what visibly goes wrong on the SOMA take (model space), as short sentences for the card: keys the model missed,
    feet under the floor, flights that break gravity, joints bent past what a body can do, legs through the obstacle,
    palms that never reach their plant"""
    out = []
    miss = meta.get("key_max_miss_cm") or []
    bad = [(f, m) for f, m in zip(meta.get("key_frames") or [], miss) if m > 10]
    if bad:
        out.append("keys missed: " + ", ".join(f"frame {f} by {m:.0f} cm" for f, m in bad) +
                   " (the keys fight each other or the run's momentum; move or re-time them)")
    W = [fk(p) for p in poses]
    soles = [min(min(w[JI[s + "ToeEnd"], 1], w[JI[s + "Foot"], 1] - 0.068) for s in ("Left", "Right")) for w in W]
    sunk = [f for f, y in enumerate(soles) if y < -0.03]
    if sunk:
        out.append(f"feet under the floor on {len(sunk)} frames (worst {100 * min(soles):.0f} cm, frame {int(np.argmin(soles))})")
    out += flights(W, soles, {f for hd in meta.get("hand_targets") or [] for f in range(hd["from"], hd["to"] + 1)})
    ref, seen = fk(REF), {}
    for f, w in enumerate(W):
        for joint, what in anatomy(w, ref):
            seen.setdefault((joint, what.split(" ")[0]), (f, what, []))[2].append(f)
    for (joint, _), (f, what, fs) in seen.items():
        out.append(f"{joint} {what} on {len(fs)} frames (from frame {f}): not something a body can do")
    ob = meta.get("obstacle")
    if ob:
        a, d, h, r = ob["near_z"], ob["depth"], ob["height"], 0.05
        hit = []
        for f, w in enumerate(W):
            for s in ("Left", "Right"):
                for n in ("Shin", "Foot", "ToeEnd"):
                    x, y, z = w[JI[s + n], :3]
                    if a - r < z < a + d + r and abs(x) < 0.9 and y < h + r:
                        hit.append((f, s + n, h + r - y))
        if hit:
            f, j, pen = max(hit, key=lambda t: t[2])
            out.append(f"legs through the obstacle on {len({h_[0] for h_ in hit})} frames (worst {j} by "
                       f"{100 * pen:.0f} cm at frame {f})")
    for hd in meta.get("hand_targets") or []:
        gap = max(np.linalg.norm(W[f][JI[hd["joint"]], :3] - np.asarray(hd["pos"]))
                  for f in range(hd["from"], min(hd["to"], len(W) - 1) + 1))
        if gap > 0.05:
            out.append(f"{hd['joint']} off its target by up to {100 * gap:.0f} cm (out of reach)")
    return out


def save_take(out_dir, world_poses, contacts, meta):
    """a take in out_dir: take.npz (model-space poses, contacts L heel, L toe, R heel, R toe), take.bvh, take.json"""
    os.makedirs(out_dir, exist_ok=True)
    poses = to_model(world_poses)
    fc = np.asarray(contacts, float)
    np.savez(os.path.join(out_dir, "take.npz"), poses=np.array(poses), contacts=fc)
    write_bvh(os.path.join(out_dir, "take.bvh"), poses)
    meta = dict(meta, frames=len(poses), seconds=round(len(poses) / FPS, 3), foot_contacts=fc.round(3).tolist())
    meta["warnings"] = checks(poses, meta)
    with open(os.path.join(out_dir, "take.json"), "w") as fh:
        json.dump(meta, fh)
    return {k: v for k, v in meta.items() if k != "foot_contacts"}


def save_keys(out_dir, r):
    """a recipe's key poses next to its take (keys.json, model space, [77, 7] each) for the pose editor"""
    rows = lambda p: np.round(p, 5).tolist()
    with open(os.path.join(out_dir, "keys.json"), "w") as fh:
        json.dump({"n": int(r["n"]), "names": NAMES, "parents": PAR, "meta": r["meta"], "hands": r["hands"],
                   "keys": [{"frame": int(f), "pose": rows(p)} for f, p in r["keys"]],
                   "effectors": [{"joint": j, "frame": int(f), "pose": rows(p)} for j, f, p in r["effectors"]]}, fh)


def load_take(take_dir):
    """a saved take back as world poses (for a transition's context) and contacts"""
    z = np.load(os.path.join(take_dir, "take.npz"))
    return [to_world(p) for p in z["poses"]], list(z["contacts"])


if __name__ == "__main__":                          # self-check: python soma.py <skeleton.npz>
    import sys
    load_skeleton(sys.argv[1])
    for nm in POSES:
        p = build(nm)
        W = fk(p)
        toe = min(W[JI["LeftToeEnd"], 1], W[JI["RightToeEnd"], 1])
        print(f"{nm:12s} heading {heading(p):6.1f} hips {W[0, :3].round(3)} toe y {toe:.3f}")
        assert abs(heading(p)) < 12.0, nm
    a = build({"base": "ready", "turns": {"Head": [["y", 20]]}, "root": {"forward": -0.5}})
    assert abs(fk(a)[0, 2] - fk(build("ready"))[0, 2] + 0.5) < 1e-6
    hl, hr = fk(build("hit_left")), fk(build("hit_right"))
    assert abs(hl[JI["Head"], 0] + hr[JI["Head"], 0]) < 0.02          # mirrored about the mid plane
    print("ok")
