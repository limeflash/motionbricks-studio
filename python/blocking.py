"""The blocking preview: a recipe's key poses interpolated frame by frame, no model and no GPU, so the poses can be
judged on the freak before Kimodo runs through them. One shot, run with the MotionBricks venv's python (numpy, scipy):
    python blocking.py <recipe> '<json args with "out">'      -> prints the take's meta as JSON"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import recipes as Rc  # noqa: E402
import soma as S  # noqa: E402


def interpolate(n, keys, root_path):
    """rotations slerped between neighbouring keys (linear: blocking, not a spline), the hips' height between the keys'
    and their ground spot along the root path; before the first key and after the last, held"""
    keys = sorted(keys, key=lambda k: k[0])
    rp = np.asarray(root_path, float)
    out = []
    for f in range(n):
        i = max([j for j, (kf, _) in enumerate(keys) if kf <= f], default=0)
        fa, pa = keys[i]
        fb, pb = keys[min(i + 1, len(keys) - 1)]
        t = 0.0 if fb == fa or f <= fa else min((f - fa) / (fb - fa), 1.0)
        p = S.blend(pa, pb, t)
        p[0, 0] = np.interp(f, rp[:, 0], rp[:, 1])
        p[0, 2] = np.interp(f, rp[:, 0], rp[:, 2])
        low = Rc.sole(p)
        if low < 0.004:                             # blended between two stances the feet would dip under the floor
            p[0, 1] += 0.004 - low
        out.append(p)
    return out


def contacts(poses):
    """a foot planted while its heel or toe is on the floor (L heel, L toe, R heel, R toe, as the models report)"""
    fc = []
    for p in poses:
        W = S.fk(p)
        row = []
        for s in ("Left", "Right"):
            row += [float(W[S.JI[s + "Foot"], 1] - 0.068 < 0.03), float(W[S.JI[s + "ToeEnd"], 1] < 0.03)]
        fc.append(row)
    return fc


def main():
    S.load_skeleton(os.environ["MBS_SKELETON"])
    args = json.loads(sys.argv[2])
    out_dir = args.pop("out")
    args.pop("seed", None)
    r = getattr(Rc, sys.argv[1])(**args)
    poses = Rc.plant(interpolate(r["n"], r["keys"], r["root_path"]), r["hands"])
    meta = dict(r["meta"], engine="blocking: keys interpolated, no model", key_frames=[int(f) for f, _ in r["keys"]])
    print(json.dumps(S.save_take(out_dir, [S.to_world(p) for p in poses], contacts(poses), meta)))


if __name__ == "__main__":
    main()
