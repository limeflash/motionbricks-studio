---
name: motion-studio
description: Turn a description of a Build a Freak character move (a hit reaction, knock-back, vault, limp, transition, idle draft) into MotionBricks / Kimodo constraints with the motionbricks-studio tools, judge the card, and export the clip JSON. Use when the owner asks for generated motion for a freak; not for strikes, which stay hand-keyed in Blender.
---

# Motion studio

You are the text layer: Kimodo runs text-free. A description goes in; key poses, frames, hand/foot targets and the
root's path come out as tool arguments.

## Rules (the owner's)
- For reactions, knock-backs, vaults, a limp, transitions and idle drafts. **Never strikes**: punches and other attacks
  stay hand-keyed in Blender (the owner rejected generated punches).
- **Our own poses only.** Start from `motion_poses` (authored from the SOMA T-pose) and your own specs. Never BONES sample
  motions, poses taken from them, or MotionBricks' Smart Primitives / locomotion modes (their targets are BONES samples).
- **Smooth.** No snapped time-warps (the `phase_warp` retiming jittered and was rejected). Let the models keep their own
  pace; slow a segment with larger token counts or later key frames instead.
- **GPU.** One job at a time; the tools refuse while Roblox or a Blender is open. Ask the owner, then pass `gpu_ok=true`.

## Which tool
| Want | Tool |
|---|---|
| hit, stagger, knock-back | `motion_hit_reaction` (front knockback = the probe's approved one) |
| get from a pose or a clip's end to another pose, turn on the spot, recover, idle draft | `motion_transition` |
| a move through your own key poses with hand/foot targets | `motion_from_keys` |
| vault an obstacle | `motion_vault` (a running jump; `style: hand` for a one-hand vault) |
| walk/run a path, limp | `motion_locomotion` (style `limp_right` / `limp_left` or your `style_pose`) |
| another style preset / travel mode on a clip | `motion_retarget` |
| the game's clip JSON | `motion_export` |

## Writing poses
Model space on the SOMA body: x = its left, y = up, z = forward, meters (leg 0.857 m; 1 stud ≈ 0.255 m). A spec extends
a library pose: `{"base": "ready", "legs": [thigh, shin, root_pitch], "turns": {"Chest": [["x", 12]]}, "ik": [...],
"root": {"forward": -0.5, "yaw": 30}}`. Turns are world axes in order: x+ pitches a trunk forward (swings a hanging limb
back), y+ turns left, z+ rolls the left side up. Read the joint positions `motion_poses` returns before placing IK
targets.

Kimodo holds keys best 8+ frames apart and misses fast ones; keep keys physically possible (feet on the floor, reachable
hands). Its result reports how far each key came out (`key_max_miss_cm`); over ~10 cm means the keys fight each other.

## Block first, then generate
Pose by pose, like an animator: write the keys, run the tool with `blocking=true` (no GPU, seconds), look at each key on
the card (the K buttons) and read its checks; fix the poses until it is clean, then run it for real. Place feet and hands
by position (`ik` targets, like IK controls) rather than by joint angles: picked by angle, a swinging foot sweeps through
whatever is below it. Remember the freak's proportions: SKULL's shoulders sit high over a big torso and his feet are tall,
so he needs more lean to reach low and more height to clear an obstacle than the SOMA body.

## After a generation
If the card says "Something looks wrong", tell the owner what and what you will change; don't present it as done.
The card plays the clip on the freak (side / ¾ / front, ¼× to 1×, the planted feet lit). Keep your reply short: the card
shows the parameters. Export only what the owner approves; the export goes to
`D:\Pictures\build a freak\forms\skull_anim\generated\` and feeds `art/freaks/skull/scripts/clips_export.py`.
