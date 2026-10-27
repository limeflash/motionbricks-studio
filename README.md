# MotionBricks Studio

A Claude Code plugin for Build a Freak: character motion made offline with NVIDIA **MotionBricks** and **Kimodo**, put
on a freak through the art pipeline's Blender retarget, shown in a card that plays the clip, and exported as the game's
clip JSON. One Go binary (MCP over stdio, a local preview server, the card embedded); two Python workers on the runtimes
already installed on H:. Its shape follows [tripo-studio-mcp](../tripo-studio-mcp).

**For** reactions, knock-backs, vaults, a limp, transitions and idle drafts. **Not for strikes**: attacks stay
hand-keyed in Blender (the owner rejected generated punches). **Our own poses only**: every pose is authored here from
the SOMA T-pose; BONES sample motions and poses (MotionBricks' Smart Primitives and locomotion modes) are never used.
**Text-free**: Llama / LLM2Vec are cut out and never downloaded (Kimodo runs with `HF_HUB_OFFLINE=1`); the calling
Claude is the text layer and turns a description into key poses, targets and a root path.

## Tools
| Tool | Engine | |
|---|---|---|
| `motion_hit_reaction` | MotionBricks, ~0.3 s | light / medium / knockback from front, back, left, right. Front knockback = the probe's approved one |
| `motion_transition` | MotionBricks, ~4 ms a step | from a pose or a clip's end through target poses (or another clip's frame) |
| `motion_from_keys` | Kimodo, ~2-3 s | full-body key poses, hand/foot targets, the root's ground path |
| `motion_vault` | Kimodo | over an obstacle by height and speed: a running jump (default, as the probe's approved one) or a one-hand vault |
| `motion_locomotion` | Kimodo | along a path; `limp_right` / `limp_left` or your own style pose (draft) |
| `motion_retarget` | Blender, ~10 s | the clip on the freak again with another style preset / travel mode |
| `motion_export` | | the loco clip JSON to `D:\Pictures\build a freak\forms\skull_anim\generated\skull_<name>_clip.json` |
| `motion_show`, `motion_poses` | | a clip's card or the recent list; the pose library |

Every generation tool takes `name`, `freak` / `form` (SKULL heavy, with the game's 4.0824 legs, is the one wired up),
and `gpu_ok`. `motion_vault` and `motion_from_keys` also take `blocking=true`: the key poses interpolated on the freak with
no model and no GPU (`python/blocking.py`, a few seconds), to judge the poses before Kimodo runs through them.

## Checks: the card says what looks wrong
Every clip is checked and the card lists what fails (a yellow "Check" pill), and the tool text tells the calling Claude to
pass it on: keys the model missed by over 10 cm, feet under the floor, legs through the obstacle, a hand or foot off its
target, a flight that breaks gravity (with both feet off and no hand planted for T s the hips must rise g T²/8 over the
line between take-off and landing: more hovers, less floats down), joints past what a body can do (a knee past 155° or an
elbow past 155°, either bent backwards or sideways off its hinge; a hip flexed past 145°, stretched back past 35°, spread
past 75° or crossed past 30°; calibrated so Kimodo's own leaps pass) (on SOMA, `soma.checks`); and on the freak itself (`blender/retarget.py`): knees and soles through the obstacle
(SKULL's feet are tall: his ankle sits 0.89 stud over the sole), a hand off its plant, anim's arm clearance pushing the
arms far off the body, the hips pulled down by the leg solve (a planted foot the leg can't reach: the freak kneels).
The card also shows each key pose as a button (frame, and how far the motion came out from it).

## The pose editor
A clip made from key poses (`motion_vault`, `motion_from_keys`, and runs of the editor itself) saves them as `keys.json`
next to its take, and its card gets an "Edit the poses" button: `http://127.0.0.1:5191/edit/<id>`. The keys stand along
the hips' path on the SOMA body (Kimodo's own skin, `somaskel77/skin_standard.npz`; built once into
`soma_skin.json`): drag the hands, feet and hips like IK controls (the feet stay planted while the body moves), turn
any joint with the rings, hide keys or show only the picked ones. "Blocking on the freak" and "Kimodo" run the clip
again through the changed poses into `<name>_edit-…` and open its card below. Only the poses change: the frames, the
hips' path (it follows the keys) and the marks stay. The page is served by whichever plugin instance holds the port.

## How a clip is made
1. A worker generates a SOMA take (77 joints, 30 fps): `python/mb_worker.py` (MotionBricks' motion player on our
   poses, ported from the probe's `mb_generate.py`) or `python/kimodo_worker.py` (Kimodo with full-body, end-effector
   and root-2D constraints, ported from `kimodo_probe.py`). Kimodo's joints map to MotionBricks' by name. Both share
   `python/soma.py` (posing, the pose library, the CMU-named BVH writer).
2. `blender/retarget.py` puts it on the freak the way the probe's `b_mb_skull.py` did: `_pipeline/b_skull_loco.py`
   with the DUMMY pipeline's `anim.py` (retarget, leg solve, arm clearance), SKULL's hinges and style, then
   `anim_export.game_clip`. Unlike the probe's strikes, anim's input smoothing stays on and nothing is time-warped.
3. The clip JSON is the format of `forms/skull_anim/loco/README.md` (15 joints, quaternions relative to rest, Blender
   axes, studs, contacts, marks, `path`). The export passes `_pipeline/loco_export_check.py`'s dry run of the game
   export (the knockback: 5.6 KB packed, hands and feet within 0.069 studs of the JSON; `clips_export.py` allows 0.08).

The card gets the freak as a small GLB (built once per freak and form, 45k triangles, 512 px WebP base colours) and the
clip's frames in the tool result, so it needs no server and works in the phone app. The same card is at
`http://127.0.0.1:5191/clip/<id>` while the server runs.

## Rules the server keeps
- One GPU job at a time (a lock around every worker call). The tools refuse while Roblox (Studio or the player) or a
  Blender is open; `gpu_ok=true` once the owner confirms nothing is in Play or rendering.
- Workers start on first use and stop after 10 minutes idle; the server stops only processes it started.
- Data: `%LOCALAPPDATA%\MotionBricksStudio` (takes, clips, the rig GLBs, worker logs).

## Build and install
```bash
npm --prefix ui install
npm --prefix ui run build
go build -o motionbricks-studio.exe .
claude plugin marketplace add G:/projects/motionbricks-studio
claude plugin install motionbricks-studio@motionbricks-studio
```
The plugin (`plugin/`) carries the MCP server entry and the `motion-studio` skill; the server runs the exe built here,
so a rebuild takes effect in the next session. `npm --prefix ui run dev` serves a card test host on :5181 that renders
real results in a sandboxed iframe.

Paths (environment overrides): `MBS_MOTIONBRICKS` `H:\tools\motionbricks`, `MBS_KIMODO` `H:\tools\kimodo`,
`MBS_HF_HOME` `H:\tools\hf`, `MBS_BLENDER` `H:\Blender\blender.exe`, `MBS_PIPELINE`
`D:\Pictures\build a freak\_pipeline`, `MBS_GAME` `G:\projects\buildafreak`, `MBS_EXPORT_DIR`, `MBS_DATA`, `MBS_PORT`.

## State (2026-10-27)
- Done: the knockback end to end (generated from our poses, played in the card, exported and checked).
- The limp (`limp_right`) reads in numbers: the bad knee bends about half as much as the good one (mean 28° vs 47°),
  it stands 41% of the time vs 50%, the trunk leans 6° over the good leg.
- The vault (`python/recipes.py`): by default a running jump like the probe's approved MotionBricks jump, blocked from our
  own poses with the feet placed like IK controls: the run-in as real strides (contact, mid-stance, push-off with the
  foot planted, 0.13 s in the air between), the take-off, then a ballistic flight (the hips on a gravity arc whose
  time comes from the height they must reach) through the knee attack, the lead leg long over the top with the trail
  knee folding out, the lead foot reaching down as the trail knee comes round, the landing; a slowing step, the stop;
  it clears the box on SOMA and on SKULL. Kimodo through those keys makes a natural leap but strays 20-25 cm from the
  airborne keys (reported on the card). `style=hand` is a one-hand tuck vault for taller obstacles: the palm is laid on the
  top after generation on SOMA and again on SKULL (the retarget carries arm rotations, not hand positions).
- Drafts, not judged by the owner: the walk along a path, keyed motion.
- Only SKULL heavy is wired up for the retarget; other freaks and forms need their rigs in `blender/retarget.py`.
