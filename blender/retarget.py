# A SOMA take onto a Build a Freak freak, the way the 2026-10-27 probe did it (forms/skull_anim/motionbricks/scripts/
# b_mb_skull.py): the SKULL loco pipeline (_pipeline/b_skull_loco.py, run with no command so only its functions load)
# with the game's heavy legs (4.0824 = 3.402 x 1.2 along the leg), anim's retarget, leg solve and arm clearance, SKULL's
# hinges and style, then anim_export.game_clip, written as the loco clip JSON (15 joints, quaternions relative to rest,
# Blender axes, studs). Unlike the probe's strikes, anim's input smoothing and arm clearance stay on (the owner rejected
# the snapped, jittery retiming): generated motion keeps its continuous velocity.
#   blender -b --factory-startup --python retarget.py -- <job.json>
# job: take_dir, name, freak, form, preset (hit | move | raw), travel (path | root | strip), out, rig_glb (optional),
#      marks (optional, seconds), pipeline (the art _pipeline folder), game (the game repo)
import json
import os
import sys

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8"))
if (JOB["freak"], JOB["form"]) != ("skull", "heavy"):
    raise SystemExit(f"RETARGET-ERROR only skull/heavy is wired up so far, not {JOB['freak']}/{JOB['form']}")
os.environ["BAF_GAME"] = JOB["game"]
sys.argv = [sys.argv[0], "--", "noop"]
exec(open(os.path.join(JOB["pipeline"], "b_skull_loco.py"), encoding="utf-8").read())

LEG0, LEG = 3.402, 4.0824                           # Catalog.SHAPES.heavy: drawn for 3.402, played at 4.0824
DZ = LEG - LEG0
_build_character, _lf_load, _load_anim = build_character, LF.load, load_anim
TAKE = JOB["take_dir"]
META = json.load(open(os.path.join(TAKE, "take.json")))


def build_character():
    """everything SKULL stands on rides DZ higher (the legs come in below, stretched)"""
    O, R = _build_character()
    for o in list(bpy.data.objects):
        if o.parent is None:
            o.location.z += DZ
    return O, R


def _legs_stretched(key, coll):
    """the game's stretch: the leg module lengthened along the leg from its hip socket (each part's origin)"""
    objs = _lf_load(key, coll)
    if key == "skull_legs":
        for o in objs:
            if o.parent is None:
                o.scale.z *= LEG / LEG0
                o.location.z += DZ
    return objs


LF.load = _legs_stretched


def load_anim():
    A, AE = _load_anim()

    def bvh(name):                                  # the take; anything else (idle_pose's CMU capture) as before
        if name not in A._BVH:
            A._BVH[name] = A.Bvh(os.path.join(TAKE, "take.bvh") if name == "take" else os.path.join(A.MOCAP, name + ".bvh"))
        return A._BVH[name]
    A.bvh = bvh
    return A, AE


def contacts(A, m, t0):
    """feet planted where the model says (heel or toe over 0.5), sampled as retarget samples the BVH; where it says
    nothing (both feet off on a held frame) anim's own test; plants under 3 frames dropped"""
    fc = np.array(META["foot_contacts"])
    b = A.bvh("take")
    i = np.round(np.clip((t0 + np.arange(len(m)) / FPS) / b.dt, 1, b.nframes - 1.001)).astype(int)
    heur = np.stack([b.contacts("R"), b.contacts("L")], 1)[i]
    f = fc[i - 1]
    out = np.where((f.max(1) == 0)[:, None], heur, np.stack([f[:, 2:].max(1), f[:, :2].max(1)], 1) > 0.5)
    for k in range(2):
        for s_, e_ in A.runs(out[:, k]):
            if e_ - s_ < 3:
                out[s_:e_, k] = False
    m.contact = out
    return m


LOOSE = {"ShoulderL": 2, "ShoulderR": 2, "ElbowL": 3, "ElbowR": 3, "WristL": 4, "WristR": 4}


def to_freak(A, rig, p):
    """a SOMA model-space point (x left, y up, z forward, m; the take's hips start over the origin) on the freak:
    Blender axes, studs, by the retarget's own scale"""
    k = rig.leg / A.bvh("take").leg * 13.957 / 0.857
    return np.array([rig.pos0[0][0] - p[0] * k, rig.pos0[0][1] + p[2] * k, p[1] * k])


WRIST_OVER_PALM = 0.5                               # studs: SKULL's wrist joint above a palm laid flat


def arm_hinge_to(A, m, rig, side, W):
    """arm_hinge for one arm onto given wrist points W [n, 3] (the elbow toward the motion's own), as the probe's
    b_mb_skull.arm_hinge_to"""
    BI = A.BI
    P = rig.joint_world(m)
    loc = m.local()
    sh, el, wr = BI["Shoulder" + side], BI["Elbow" + side], BI["Wrist" + side]
    u0, f0 = rig.pos0[el] - rig.pos0[sh], rig.pos0[wr] - rig.pos0[el]
    S_, E = P[:, sh], P[:, el]
    q_w = m.rot[:, BI["Waist"]]
    pe = E - (S_ + W) / 2
    pe = pe / np.maximum(np.linalg.norm(pe, axis=1, keepdims=True), 1e-6)
    q_up, th, _ = hinge(A, S_, W, u0, f0, q_w, 1.0, pe, 1.0)
    loc[:, sh] = A.qmul(A.qinv(q_w), q_up)
    loc[:, el] = A.qaxis(X_, th)
    m.set_local(loc)
    return m


def plant_hands(A, m, rig):
    """the palm plants of the recipe (META hand_targets) laid on the freak: the wrist path offset onto the plant, eased
    in and out (smoothstep), the arm re-solved on its hinges. Rotations alone leave SKULL's hands short of the box"""
    for hd in META.get("hand_targets") or []:
        side = "L" if hd["joint"].startswith("Left") else "R"
        tgt = to_freak(A, rig, hd["pos"]) + np.array([0.0, 0.0, WRIST_OVER_PALM])
        W = rig.joint_world(m)[:, A.BI["Wrist" + side]].copy()
        n = len(W)
        for f in range(max(hd["in"], 0), min(hd["out"], n - 1) + 1):
            if f < hd["from"]:
                u = (f - hd["in"]) / max(hd["from"] - hd["in"], 1)
            elif f <= hd["to"]:
                u = 1.0
            else:
                u = 1 - (f - hd["to"]) / max(hd["out"] - hd["to"], 1)
            W[f] += (u * u * (3 - 2 * u)) * (tgt - W[f])
        arm_hinge_to(A, m, rig, side, W)
    return m


def freak_checks(A, rig, m):
    """what goes wrong on the freak itself: the hips pulled down by the leg solve (a pinned foot the leg can't reach:
    the freak kneels), its knees and soles through the obstacle, a wrist off its plant"""
    out = []
    P = rig.joint_world(m)
    raw = A.retarget("take", rig, 1 / FPS, None).root[:, 2]     # the hips' height the motion itself asks for
    n = min(len(raw), len(m.root))
    sink = raw[:n] - m.root[:n, 2]
    bad = np.nonzero(sink > 0.6)[0]
    if bad.size:
        out.append("on the freak: the hips sink up to %.1f studs below the motion on frames %d-%d (a planted foot the "
                   "leg can't reach: the legs fold)" % (sink.max(), bad[0] + 1, bad[-1] + 1))
    ob = META.get("obstacle")
    if ob:
        near = to_freak(A, rig, [0.0, ob["height"], ob["near_z"]])
        far = to_freak(A, rig, [0.0, ob["height"], ob["near_z"] + ob["depth"]])
        r, hits = 0.5, []                           # knees by their thickness; feet by their soles (the ankle's rest height)
        for f in range(len(P)):
            for j in ("KneeR", "KneeL", "AnkleR", "AnkleL"):
                x, y, z = P[f, A.BI[j]]
                low = z - (rig.pos0[A.BI[j]][2] if j.startswith("Ankle") else r)
                if near[1] - r < y < far[1] + r and abs(x - near[0]) < 3.5 and low < near[2]:
                    hits.append((f, j.replace("Ankle", "Foot"), near[2] - low))
        if hits:
            f, j, pen = max(hits, key=lambda t: t[2])
            out.append("on the freak: %s through the obstacle on %d frames (worst %s by %.1f studs at frame %d)"
                       % ("legs", len({h[0] for h in hits}), j, pen, f))
    for hd in META.get("hand_targets") or []:
        side = "L" if hd["joint"].startswith("Left") else "R"
        tgt = to_freak(A, rig, hd["pos"]) + np.array([0.0, 0.0, WRIST_OVER_PALM])
        gap = max(np.linalg.norm(P[f, A.BI["Wrist" + side]] - tgt) for f in range(hd["from"], min(hd["to"], len(P) - 1) + 1))
        if gap > 0.4:
            out.append("on the freak: %s misses its plant by up to %.1f studs" % (hd["joint"], gap))
    return out


def clip_fn(A, preset):
    def f(rig):
        m = contacts(A, A.face_pose(A.retarget("take", rig, 1 / FPS, None)), 1 / FPS)
        if preset == "raw":
            return m
        arm_hinge(A, m, rig)                        # SKULL's hinged arms, as the probe's hit() and turn()
        drag(A, m, LOOSE)
        if preset == "hit":
            rattle(A, m, 1.5, tau=0.13)
        else:
            rattle(A, m, 1.0)
        return m
    return f


def export_rig(path):
    """SKULL as the card's preview model: the Rig_* empties and the meshes on them (helpers left out), decimated to a
    phone-friendly size, base colours at most 512 px as WebP, no normal maps; joints keep their names and rest at zero
    rotation"""
    root = D["Rig_Hips"]
    keep = [root] + [o for o in root.children_recursive if not o.name.startswith(("Std_", "Socket_", "Muzzle"))]
    meshes = [o for o in keep if o.type == "MESH"]
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
    ratio = min(1.0, 45000 / max(tris, 1))
    for o in meshes:
        if ratio < 1.0:
            o.modifiers.new("preview", "DECIMATE").ratio = ratio
    for mat in bpy.data.materials:
        for nd in (mat.node_tree.nodes if mat.node_tree else []):
            if nd.type == "NORMAL_MAP":
                for link in list(nd.outputs[0].links):
                    mat.node_tree.links.remove(link)
    for img in bpy.data.images:
        if img.size[0] > 512 or img.size[1] > 512:
            k = 512 / max(img.size)
            img.scale(max(1, int(img.size[0] * k)), max(1, int(img.size[1] * k)))
    bpy.ops.object.select_all(action="DESELECT")
    for o in keep:
        o.select_set(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True,
                              export_animations=False, export_cameras=False, export_lights=False,
                              export_image_format="WEBP", export_image_quality=80)
    print("RIG-GLB %s %d meshes, %d tris x %.2f" % (path, len(meshes), tris, ratio), flush=True)


def main():
    for o in list(D):
        bpy.data.objects.remove(o, do_unlink=True)
    sc.render.fps = FPS
    O, R, J, parts = build_skull()
    A, AE = load_anim()
    rig = skull_rig(A, parts, J)
    name = JOB["name"]
    AE.KEY[name] = name
    if JOB["travel"] == "path":
        AE.PATH = tuple(AE.PATH) + (name,)
    elif JOB["travel"] == "strip":
        AE.STRIP = tuple(AE.STRIP) + (name,)
    m = leg_hinge(A, A.build_clip(name, clip_fn(A, JOB["preset"]), rig), rig)
    if JOB["preset"] != "raw":
        plant_hands(A, m, rig)                      # after anim's arm clearance, which would push a planted hand off
    out = write_clip(A, AE, rig, name, m, JOB["out"])
    out["checks"] = freak_checks(A, rig, m)
    if JOB.get("marks"):
        out["marks"] = dict(out.get("marks") or {}, **JOB["marks"])
    out["source"] = {k: META.get(k) for k in ("op", "engine", "frames") if k in META}
    out["freak"], out["form"], out["leg"] = JOB["freak"], JOB["form"], LEG
    out["soma_scale"] = round(rig.leg / A.bvh("take").leg * 13.957 / 0.857, 4)   # studs per SOMA meter, as retargeted
    out["note"] = ("Generated motion (motionbricks-studio) on SKULL heavy with the game's heavy legs (4.0824). Joint "
                   "rotations (w, x, y, z) relative to the rest pose in Blender axes (Z up, facing +Y, right +X); "
                   "pelvis_offset = (forward, up), pelvis_side = right: the hips' offset from their rest place, studs; "
                   "path = (right, forward) travel the game's body carries; contact bits: 1 right foot, 2 left foot; "
                   "marks in seconds ((frame - 1) / 30).")
    with open(JOB["out"], "w") as fh:
        json.dump(out, fh)
    if JOB.get("rig_glb") and not os.path.exists(JOB["rig_glb"]):
        export_rig(JOB["rig_glb"])
    print("RETARGET-OK", JOB["out"], flush=True)


main()
