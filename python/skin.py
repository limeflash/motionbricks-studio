"""The SOMA body for the pose editor: Kimodo's skin (somaskel77/skin_standard.npz: 18k vertices, 8 bone weights each) as
one JSON the page skins on the GPU, with the 4 strongest weights a vertex, the bind transforms, and each joint's world
turn in the MotionBricks reference pose: a pose's world turn times its inverse is Kimodo's global turn, the frame the
bind transforms are in (kimodo_worker.skeleton_poses does the same). One shot, run with the MotionBricks venv's python:
    python skin.py <skin_standard.npz> <out.json>"""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import soma as S  # noqa: E402


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main(src, out):
    S.load_skeleton(os.environ["MBS_SKELETON"])
    z = np.load(src)
    if [str(n) for n in z["rig_joint_names"]] != S.NAMES:
        raise SystemExit("the skin's joints are not the SOMA skeleton's")
    top = np.argsort(-z["lbs_weights"], axis=1)[:, :4]
    idx = np.take_along_axis(z["lbs_indices"], top, 1)
    w = np.take_along_axis(z["lbs_weights"], top, 1)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-8)
    with open(out, "w") as fh:
        json.dump({"vertices": b64(z["bind_vertices"].astype(np.float32)), "faces": b64(z["faces"].astype(np.uint32)),
                   "skin_index": b64(idx.astype(np.uint16)), "skin_weight": b64(w.astype(np.float32)),
                   "bind": b64(z["bind_rig_transform"].astype(np.float32)),       # [77][4][4] row-major
                   "ref_q": b64(S.fk(S.REF)[:, 3:].astype(np.float32))}, fh)      # [77] x, y, z, w


if __name__ == "__main__":
    main(*sys.argv[1:3])
