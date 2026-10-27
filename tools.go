package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// tool is one MCP tool.
type tool struct {
	Name        string
	Description string
	InputSchema map[string]any
	UI          bool // results render in the card widget
	run         func(context.Context, *app, json.RawMessage) (reply, error)
}

// handler decodes the arguments strictly, so a misspelt option is an error instead of a silently ignored setting.
func handler[T any](f func(context.Context, *app, T) (reply, error)) func(context.Context, *app, json.RawMessage) (reply, error) {
	return func(ctx context.Context, a *app, raw json.RawMessage) (reply, error) {
		var args T
		if len(raw) > 0 && string(raw) != "null" {
			d := json.NewDecoder(bytes.NewReader(raw))
			d.DisallowUnknownFields()
			if err := d.Decode(&args); err != nil {
				return reply{}, fmt.Errorf("bad arguments: %v", err)
			}
		}
		return f(ctx, a, args)
	}
}

// ---- schema helpers ----

type props = map[string]any

func object(required []string, ps ...props) map[string]any {
	all := props{}
	for _, p := range ps {
		for k, v := range p {
			all[k] = v
		}
	}
	s := map[string]any{"type": "object", "properties": all, "additionalProperties": false}
	if len(required) > 0 {
		s["required"] = required
	}
	return s
}

func str(desc string) any     { return props{"type": "string", "description": desc} }
func integer(desc string) any { return props{"type": "integer", "description": desc} }
func number(desc string) any  { return props{"type": "number", "description": desc} }
func boolean(desc string) any { return props{"type": "boolean", "description": desc} }
func anyJSON(desc string) any { return props{"description": desc} }
func list(desc string, item any) any {
	return props{"type": "array", "items": item, "description": desc}
}
func enum(desc string, values ...string) any {
	return props{"type": "string", "enum": values, "description": desc}
}

const poseDoc = `A pose spec (JSON) on the SOMA body (1.7 m, leg 0.857 m; model space: x = its left, y = up, z = forward, meters), always built from the straight-legged stand: {"base": library pose it extends (motion_poses), "legs": [thigh, shin, root_pitch] knees bent in degrees with the feet kept flat, "turns": {"Chest": [["x", 12]]} extra world-axis turns per joint in order (x+ pitches a trunk forward or swings a hanging limb back, y+ turns left, z+ rolls the left side up), "shift": [left, up, forward] root offset, "ik": [{"joint": "LeftHand", "target": [x, y, z], "pole": [x, y, z]}], "mirror": true, "root": {"forward", "left", "up" (m), "yaw" (deg left)} where the body has travelled to}. A plain string names a library pose.`

// what every generation tool takes besides its own parameters
type common struct {
	Name  string `json:"name"`
	Freak string `json:"freak"`
	Form  string `json:"form"`
	GPUOK bool   `json:"gpu_ok"`
}

var commonProps = props{
	"name":   str("Clip name (letters, digits, _), the game key and the export file's name."),
	"freak":  enum("Freak to put the clip on (default skull).", "skull"),
	"form":   enum("Body form (default heavy).", "heavy"),
	"gpu_ok": boolean("Set only after the owner confirmed Roblox is not in Play and no Blender render runs: skips the check that refuses while Roblox (Studio or the player) or a Blender is open."),
}

var nameRe = regexp.MustCompile(`[^a-z0-9_]+`)

func (c *common) fill(defName string) {
	c.Name = strings.Trim(nameRe.ReplaceAllString(strings.ToLower(c.Name), "_"), "_")
	if c.Name == "" {
		c.Name = defName
	}
	if c.Freak == "" {
		c.Freak = "skull"
	}
	if c.Form == "" {
		c.Form = "heavy"
	}
}

// ---- records ----

// record is one generated clip: data/takes/<id>/record.json next to the take and the retargeted clip.
type record struct {
	ID       string         `json:"id"`
	Name     string         `json:"name"`
	Tool     string         `json:"tool"`
	Created  string         `json:"created"`
	Params   map[string]any `json:"params"`
	Take     map[string]any `json:"take"`
	Freak    string         `json:"freak"`
	Form     string         `json:"form"`
	Preset   string         `json:"preset"`
	Travel   string         `json:"travel"`
	Marks    map[string]any `json:"marks,omitempty"`
	Exported []string       `json:"exported,omitempty"`
}

func (a *app) dir(id string) string { return filepath.Join(a.data, "takes", id) }
func (a *app) clipPath(r *record) string {
	return filepath.Join(a.dir(r.ID), fmt.Sprintf("clip_%s_%s.json", r.Freak, r.Form))
}
func (a *app) rigPath(freak, form string) string {
	return filepath.Join(a.data, "rigs", freak+"_"+form+".glb")
}

func (a *app) save(r *record) error {
	b, _ := json.MarshalIndent(r, "", "  ")
	return os.WriteFile(filepath.Join(a.dir(r.ID), "record.json"), b, 0o644)
}

func (a *app) load(id string) (*record, error) {
	if id == "" || strings.ContainsAny(id, `/\.`) {
		return nil, fmt.Errorf("no clip %q", id)
	}
	b, err := os.ReadFile(filepath.Join(a.dir(id), "record.json"))
	if err != nil {
		return nil, fmt.Errorf("no clip %q (motion_show lists them)", id)
	}
	var r record
	return &r, json.Unmarshal(b, &r)
}

func (a *app) recent(n int) []*record {
	ents, _ := os.ReadDir(filepath.Join(a.data, "takes"))
	var out []*record
	for _, e := range ents {
		if r, err := a.load(e.Name()); err == nil {
			out = append(out, r)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Created > out[j].Created })
	if len(out) > n {
		out = out[:n]
	}
	return out
}

func asMap(v any) map[string]any {
	b, _ := json.Marshal(v)
	var m map[string]any
	json.Unmarshal(b, &m)
	delete(m, "gpu_ok")
	return m
}

// generate runs one worker op into a new take, retargets it and returns the clip's card.
func (a *app) generate(ctx context.Context, w *worker, c common, toolName, op string, opArgs map[string]any, params any,
	preset, travel string, marks func(take map[string]any) map[string]any) (reply, error) {
	id := c.Name + "-" + time.Now().Format("0102-150405")
	opArgs["out"] = a.dir(id)
	take, err := a.gpuJob(ctx, w, c.GPUOK, op, opArgs)
	if err != nil {
		return reply{}, err
	}
	r := &record{ID: id, Name: c.Name, Tool: toolName, Created: time.Now().Format(time.RFC3339), Params: asMap(params),
		Take: take, Freak: c.Freak, Form: c.Form, Preset: preset, Travel: travel}
	if marks != nil {
		r.Marks = marks(take)
	}
	if err := a.save(r); err != nil {
		return reply{}, err
	}
	if err := a.retargetRecord(ctx, r); err != nil {
		return reply{Text: fmt.Sprintf("Generated take %s (%v frames), but the retarget failed.", id, take["frames"])}, err
	}
	return a.clipReply(r)
}

func (a *app) retargetRecord(ctx context.Context, r *record) error {
	return a.retarget(ctx, map[string]any{"take_dir": a.dir(r.ID), "name": r.Name, "freak": r.Freak, "form": r.Form,
		"preset": r.Preset, "travel": r.Travel, "out": a.clipPath(r), "rig_glb": a.rigPath(r.Freak, r.Form),
		"marks": r.Marks})
}

// takeDir resolves a clip id given by the caller to its take folder (for transitions from or to it).
func (a *app) takeDir(id string) (string, error) {
	if _, err := a.load(id); err != nil {
		return "", err
	}
	return a.dir(id), nil
}

// ---- the tools ----

type poseArgs struct {
	GPUOK bool `json:"gpu_ok"`
}

type hitArgs struct {
	common
	Direction     string   `json:"direction"`
	Strength      string   `json:"strength"`
	DistanceStuds *float64 `json:"distance_studs"`
	Land          string   `json:"land"`
}

type step struct {
	Targets []json.RawMessage `json:"targets"`
	Tokens  []int             `json:"tokens,omitempty"`
	Early   int               `json:"early,omitempty"`
}

type transitionArgs struct {
	common
	Start    json.RawMessage   `json:"start"`
	FromClip string            `json:"from_clip"`
	KeepFrom *int              `json:"keep_from"`
	Context  []json.RawMessage `json:"context"`
	Steps    []step            `json:"steps"`
	Hold     *int              `json:"hold"`
	Travel   string            `json:"travel"`
}

type keyArgs struct {
	common
	Frames    int               `json:"frames"`
	Keys      []json.RawMessage `json:"keys"`
	Effectors []json.RawMessage `json:"effectors"`
	RootPath  json.RawMessage   `json:"root_path"`
	Seed      int               `json:"seed"`
	Steps     int               `json:"steps"`
	Travel    string            `json:"travel"`
}

type vaultArgs struct {
	common
	HeightStuds   float64 `json:"height_studs"`
	Speed         string  `json:"speed"`
	ApproachStuds float64 `json:"approach_studs"`
	Seed          int     `json:"seed"`
}

type locoArgs struct {
	common
	Path       [][2]float64    `json:"path"`
	SpeedStuds float64         `json:"speed_studs"`
	Style      string          `json:"style"`
	StylePose  json.RawMessage `json:"style_pose"`
	Seed       int             `json:"seed"`
}

type retargetArgs struct {
	Clip   string `json:"clip"`
	Preset string `json:"preset"`
	Travel string `json:"travel"`
}

type exportArgs struct {
	Clip      string `json:"clip"`
	Name      string `json:"name"`
	OutDir    string `json:"out_dir"`
	Overwrite bool   `json:"overwrite"`
}

type showArgs struct {
	Clip string `json:"clip"`
}

func or(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

// clipRef turns {"clip": id, "frame": n, "root": {...}} into the worker's {"take": dir, ...}; specs pass through.
func (a *app) clipRef(raw json.RawMessage) (any, error) {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, err
	}
	if m, ok := v.(map[string]any); ok {
		if id, ok := m["clip"].(string); ok {
			dir, err := a.takeDir(id)
			if err != nil {
				return nil, err
			}
			delete(m, "clip")
			m["take"] = dir
		}
	}
	return v, nil
}

func raws(rs []json.RawMessage) []any {
	out := make([]any, 0, len(rs))
	for _, r := range rs {
		var v any
		json.Unmarshal(r, &v)
		out = append(out, v)
	}
	return out
}

func secs(take map[string]any, key string) any {
	if f, ok := take[key].(float64); ok {
		return math.Round(f/30*1e4) / 1e4
	}
	return nil
}

var tools = []tool{
	{Name: "motion_poses", Description: "The pose library (our own poses, authored from the SOMA T-pose; never BONES samples): each pose's spec, what it is, and where its hips, head, hands and feet are (model space, m). Read it before writing pose specs. " + poseDoc,
		InputSchema: object(nil, props{"gpu_ok": commonProps["gpu_ok"]}),
		run: handler(func(ctx context.Context, a *app, p poseArgs) (reply, error) {
			r, err := a.gpuJob(ctx, a.mb, p.GPUOK, "poses", map[string]any{})
			if err != nil {
				return reply{}, err
			}
			b, _ := json.MarshalIndent(r["poses"], "", " ")
			return say("Pose library (model space: x left, y up, z forward, meters; 1 stud ≈ 0.255 m on SOMA):\n" + string(b)), nil
		})},

	{Name: "motion_hit_reaction", UI: true,
		Description: "Hit reaction or knock-back (MotionBricks, ~0.3 s): the blow already landing as context, then the body recovering to a ready stand, knocked back away from the hit. strength knockback with direction front is the probe's knockback the owner liked (2026-10-27). Retargeted onto the freak; returns the clip card.",
		InputSchema: object(nil, commonProps, props{
			"direction":      enum("Where the blow comes from (default front: the body is knocked backwards).", "front", "back", "left", "right"),
			"strength":       enum("light: a flinch on the spot; medium: a stagger; knockback (default): thrown back, lands in a crouch, recovers.", "light", "medium", "knockback"),
			"distance_studs": number("How far the body travels (studs; defaults 1.4 / 1.6 / 3.5)."),
			"land":           enum("knockback only: crouch (default) or stand (no crouch on landing).", "crouch", "stand"),
		}),
		run: handler(func(ctx context.Context, a *app, p hitArgs) (reply, error) {
			p.fill("hit_" + or(p.Strength, "knockback"))
			args := map[string]any{"direction": or(p.Direction, "front"), "strength": or(p.Strength, "knockback"), "land": or(p.Land, "crouch")}
			if p.DistanceStuds != nil {
				args["distance"] = *p.DistanceStuds * stud
			}
			return a.generate(ctx, a.mb, p.common, "motion_hit_reaction", "hit_reaction", args, p, "hit", "path",
				func(t map[string]any) map[string]any {
					m := map[string]any{"impact": secs(t, "impact_frame")}
					if sf, ok := t["step_frames"].([]any); ok && len(sf) > 0 {
						m["recovered"] = secs(map[string]any{"f": sf[len(sf)-1]}, "f")
						if len(sf) > 1 {
							m["land"] = secs(map[string]any{"f": sf[0]}, "f")
						}
					}
					return m
				})
		})},

	{Name: "motion_transition", UI: true,
		Description: "Transition or inbetween (MotionBricks, ~4 ms a step): from a pose or the end of a clip, through one or more targets (pose specs, or a frame of another clip as {\"clip\": id, \"frame\": n}, stood where the motion ends). Each step's segment is the model's own pace (6-8 tokens is its shortest, ~0.6 s). Use for turns on the spot, recoveries, idle drafts, getting from one clip into another. " + poseDoc,
		InputSchema: object([]string{"steps"}, commonProps, props{
			"start":     anyJSON("Start pose spec (default \"ready\"). Ignored with from_clip."),
			"from_clip": str("Start from the end of this clip id instead: its frames are recorded as the motion so far and kept in the result."),
			"keep_from": integer("With from_clip: how many of its last frames to keep (default all)."),
			"context":   list("Poses recorded as already shown before the first step (an impact already landing, say).", props{}),
			"steps": list("In order: {\"targets\": [pose spec or {\"clip\", \"frame\", \"root\"}], \"tokens\": allowed token counts (each ~0.1 s; e.g. [6,7,8] quick, [10,11,12,13] slow), \"early\": frames to hand over early}.",
				object([]string{"targets"}, props{"targets": list("Target poses.", props{}), "tokens": list("Allowed token counts.", props{"type": "integer"}), "early": integer("Frames early.")})),
			"hold":   integer("Frames the last pose is held (default 12)."),
			"travel": enum("Where the clip's travel goes in the export: path (default; the game carries the body), root (kept in the hips) or strip (in place).", "path", "root", "strip"),
		}),
		run: handler(func(ctx context.Context, a *app, p transitionArgs) (reply, error) {
			p.fill("transition")
			args := map[string]any{"context": raws(p.Context)}
			if len(p.Start) > 0 {
				args["start"] = raws([]json.RawMessage{p.Start})[0]
			}
			if p.FromClip != "" {
				dir, err := a.takeDir(p.FromClip)
				if err != nil {
					return reply{}, err
				}
				args["from_take"] = dir
				if p.KeepFrom != nil {
					args["keep_from"] = *p.KeepFrom
				}
			}
			if p.Hold != nil {
				args["hold"] = *p.Hold
			}
			var steps []map[string]any
			for _, s := range p.Steps {
				var tg []any
				for _, t := range s.Targets {
					v, err := a.clipRef(t)
					if err != nil {
						return reply{}, err
					}
					tg = append(tg, v)
				}
				st := map[string]any{"targets": tg, "early": s.Early}
				if len(s.Tokens) > 0 {
					st["tokens"] = s.Tokens
				}
				steps = append(steps, st)
			}
			args["steps"] = steps
			return a.generate(ctx, a.mb, p.common, "motion_transition", "sequence", args, p, "move", or(p.Travel, "path"), nil)
		})},

	{Name: "motion_from_keys", UI: true,
		Description: "Motion through key poses (Kimodo, text-free, ~3 s): full-body key poses at frames, hand/foot targets at frames, and the hips' ground path. Kimodo fills in the rest smoothly. Keys hold best spaced 8+ frames apart. Not for strikes. " + poseDoc,
		InputSchema: object([]string{"frames", "keys"}, commonProps, props{
			"frames":    integer("Clip length in frames at 30 fps (max 300)."),
			"keys":      list("Full-body keys: {\"frame\": n (0-based), \"pose\": pose spec}. Its \"root\" places the body on the floor.", object([]string{"frame", "pose"}, props{"frame": integer("Frame."), "pose": anyJSON("Pose spec.")})),
			"effectors": list("End-effector targets: {\"joint\": LeftHand | RightHand | LeftFoot | RightFoot, \"frame\": n, \"position\": [x, y, z] model space, m}.", object([]string{"joint", "frame", "position"}, props{"joint": str("Joint."), "frame": integer("Frame."), "position": list("x, y, z (m).", props{"type": "number"})})),
			"root_path": anyJSON("The hips' ground path: \"keys\" (default: straight lines between the keys' hips), \"none\", or [[frame, left, forward], ...] in meters."),
			"seed":      integer("Random seed (default 0); change it for a variation."),
			"steps":     integer("Diffusion steps (default 100)."),
			"travel":    enum("Export travel: path (default), root or strip.", "path", "root", "strip"),
		}),
		run: handler(func(ctx context.Context, a *app, p keyArgs) (reply, error) {
			p.fill("keyed")
			args := map[string]any{"n": p.Frames, "keys": raws(p.Keys), "effectors": raws(p.Effectors), "seed": p.Seed}
			if p.Steps > 0 {
				args["steps"] = p.Steps
			}
			if len(p.RootPath) > 0 {
				args["root_path"] = raws([]json.RawMessage{p.RootPath})[0]
			}
			return a.generate(ctx, a.kimodo, p.common, "motion_from_keys", "generate", args, p, "move", or(p.Travel, "path"), nil)
		})},

	{Name: "motion_vault", UI: true,
		Description: "Vault over an obstacle (Kimodo through our own vault keys: approach, hands planted on the top, legs tucked over, landing, run-out; text-free). Draft quality: judge it on the card.",
		InputSchema: object(nil, commonProps, props{
			"height_studs":   number("Obstacle height (studs, default 2.5)."),
			"speed":          enum("Approach speed (default run).", "walk", "run"),
			"approach_studs": number("Run-up before the obstacle (studs, default 6)."),
			"seed":           integer("Random seed (default 0)."),
		}),
		run: handler(func(ctx context.Context, a *app, p vaultArgs) (reply, error) {
			p.fill("vault")
			args := map[string]any{"height": def(p.HeightStuds, 2.5) * stud, "speed": or(p.Speed, "run"),
				"approach": def(p.ApproachStuds, 6) * stud, "seed": p.Seed}
			return a.generate(ctx, a.kimodo, p.common, "motion_vault", "vault", args, p, "move", "path",
				func(t map[string]any) map[string]any {
					return map[string]any{"takeoff": secs(t, "takeoff_frame"), "land": secs(t, "land_frame")}
				})
		})},

	{Name: "motion_locomotion", UI: true,
		Description: "Walk or run along a path (Kimodo, root path only, text-free), optionally in a style from our own key pose: style limp_right / limp_left repeats a limp pose (the bad leg stiff, the trunk over the good one) each stride, or style_pose gives your own pose spec for it.",
		InputSchema: object([]string{"path"}, commonProps, props{
			"path":        list("Points [forward, left] in studs from the start; the body starts at [0, 0] facing forward.", list("", props{"type": "number"})),
			"speed_studs": number("Speed (studs/s, default 6: a walk; 13 is a run)."),
			"style":       enum("Gait style (default none).", "none", "limp_right", "limp_left"),
			"style_pose":  anyJSON("Your own style pose spec, repeated each stride on the right foot's stance (instead of style)."),
			"seed":        integer("Random seed (default 0)."),
		}),
		run: handler(func(ctx context.Context, a *app, p locoArgs) (reply, error) {
			p.fill("locomotion")
			if len(p.Path) == 0 {
				return reply{}, fmt.Errorf("bad arguments: path needs at least one point")
			}
			pts := make([][2]float64, len(p.Path))
			for i, q := range p.Path {
				pts[i] = [2]float64{q[0] * stud, q[1] * stud}
			}
			args := map[string]any{"path": pts, "speed": def(p.SpeedStuds, 6) * stud, "style": or(p.Style, "none"), "seed": p.Seed}
			if len(p.StylePose) > 0 {
				args["style_pose"] = raws([]json.RawMessage{p.StylePose})[0]
			}
			return a.generate(ctx, a.kimodo, p.common, "motion_locomotion", "locomotion", args, p, "move", "path", nil)
		})},

	{Name: "motion_retarget", UI: true,
		Description: "Put an existing clip's motion on the freak again with another style preset or travel mode (Blender, ~10 s, no GPU).",
		InputSchema: object([]string{"clip"}, props{
			"clip":   str("Clip id."),
			"preset": enum("hit: SKULL's hinged arms, loose follow-through, a strong bone rattle; move: the same with a light rattle; raw: the retarget alone.", "hit", "move", "raw"),
			"travel": enum("path, root or strip.", "path", "root", "strip"),
		}),
		run: handler(func(ctx context.Context, a *app, p retargetArgs) (reply, error) {
			r, err := a.load(p.Clip)
			if err != nil {
				return reply{}, err
			}
			r.Preset, r.Travel = or(p.Preset, r.Preset), or(p.Travel, r.Travel)
			if err := a.retargetRecord(ctx, r); err != nil {
				return reply{}, err
			}
			a.save(r)
			return a.clipReply(r)
		})},

	{Name: "motion_export", UI: true,
		Description: "Write a clip as the game's clip JSON (the loco format: 15 joints, quaternions relative to rest, Blender axes, studs, contacts, marks, path) to <out_dir>/<freak>_<name>_clip.json, for art/freaks/skull/scripts/clips_export.py.",
		InputSchema: object([]string{"clip"}, props{
			"clip":      str("Clip id."),
			"name":      str("File name part (default the clip's name)."),
			"out_dir":   str(`Folder (default D:\Pictures\build a freak\forms\skull_anim\generated).`),
			"overwrite": boolean("Replace an existing file."),
		}),
		run: handler(func(ctx context.Context, a *app, p exportArgs) (reply, error) {
			r, err := a.load(p.Clip)
			if err != nil {
				return reply{}, err
			}
			b, err := os.ReadFile(a.clipPath(r))
			if err != nil {
				return reply{}, err
			}
			name := strings.Trim(nameRe.ReplaceAllString(strings.ToLower(or(p.Name, r.Name)), "_"), "_")
			out := filepath.Join(or(p.OutDir, a.exportDir), fmt.Sprintf("%s_%s_clip.json", r.Freak, name))
			if _, err := os.Stat(out); err == nil && !p.Overwrite {
				return reply{}, fmt.Errorf("%s exists; pass overwrite=true or another name", out)
			}
			if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
				return reply{}, err
			}
			if err := os.WriteFile(out, b, 0o644); err != nil {
				return reply{}, err
			}
			r.Exported = append(r.Exported, out)
			a.save(r)
			rep, err := a.clipReply(r)
			rep.Text = "Exported " + out + "\n" + rep.Text
			return rep, err
		})},

	{Name: "motion_show", UI: true,
		Description: "Show a clip's card again, or with no clip list the recent clips.",
		InputSchema: object(nil, props{"clip": str("Clip id (omit to list).")}),
		run: handler(func(ctx context.Context, a *app, p showArgs) (reply, error) {
			if p.Clip != "" {
				r, err := a.load(p.Clip)
				if err != nil {
					return reply{}, err
				}
				return a.clipReply(r)
			}
			rs := a.recent(30)
			if len(rs) == 0 {
				return say("No clips yet."), nil
			}
			var b strings.Builder
			for _, r := range rs {
				fmt.Fprintf(&b, "%s  %s  %s/%s  %v frames%s\n", r.ID, r.Tool, r.Freak, r.Form, r.Take["frames"], map[bool]string{true: "  exported", false: ""}[len(r.Exported) > 0])
			}
			rep, err := a.clipReply(rs[0])
			rep.Text = b.String() + "\nLatest:\n" + rep.Text
			return rep, err
		})},
}

const stud = 0.857 / 3.358 // one SKULL stud on SOMA, by leg length (the probe's convention)

func def(v, d float64) float64 {
	if v == 0 {
		return d
	}
	return v
}

// ---- replies and the card ----

type reply struct {
	Text string
	Card map[string]any
	Meta map[string]any
}

func say(s string) reply { return reply{Text: s} }

func (r reply) result(t *tool, err error) map[string]any {
	if err != nil {
		r.Text = strings.TrimSpace(r.Text + "\nError: " + err.Error())
		if t != nil && t.UI && r.Card == nil {
			r.Card = map[string]any{"kind": "error", "title": t.Name}
		}
		if r.Card != nil {
			r.Card["error"] = err.Error()
		}
	}
	res := map[string]any{"content": []map[string]any{{"type": "text", "text": r.Text}}, "isError": err != nil}
	if r.Card != nil {
		r.Card["text"] = r.Text // Claude Code shows the model structuredContent rather than content when both exist
		res["structuredContent"] = r.Card
	}
	if r.Meta != nil {
		res["_meta"] = map[string]any{"motion": r.Meta}
	}
	return res
}

// clip JSON keys -> the preview rig's joint nodes
var rigNodes = [][2]string{{"hips", "Rig_Hips"}, {"waist", "Rig_Waist"}, {"neck", "Rig_Neck"},
	{"shR", "Rig_ShoulderR"}, {"elR", "Rig_ElbowR"}, {"wrR", "Rig_WristR"}, {"shL", "Rig_ShoulderL"}, {"elL", "Rig_ElbowL"},
	{"wrL", "Rig_WristL"}, {"hipR", "Rig_HipJR"}, {"kneeR", "Rig_KneeR"}, {"ankleR", "Rig_AnkleR"},
	{"hipL", "Rig_HipJL"}, {"kneeL", "Rig_KneeL"}, {"ankleL", "Rig_AnkleL"}}

func (a *app) clipReply(r *record) (reply, error) {
	b, err := os.ReadFile(a.clipPath(r))
	if err != nil {
		return reply{}, fmt.Errorf("clip %s has no %s/%s clip yet: %v", r.ID, r.Freak, r.Form, err)
	}
	var c struct {
		FPS     int                                   `json:"fps"`
		Frames  int                                   `json:"frames"`
		Loop    bool                                  `json:"loop"`
		Marks   map[string]any                        `json:"marks"`
		Contact []int                                 `json:"contact"`
		Scale   float64                               `json:"soma_scale"`
		Joints  map[string]map[string]json.RawMessage `json:"joints"`
	}
	if err := json.Unmarshal(b, &c); err != nil {
		return reply{}, err
	}
	q := make([][][4]float64, c.Frames)
	off := make([][3]float64, c.Frames) // right, forward, up (Blender axes, studs) incl. the path
	for f := 0; f < c.Frames; f++ {
		row := c.Joints[fmt.Sprint(f+1)]
		for _, jn := range rigNodes {
			var v [4]float64
			json.Unmarshal(row[jn[0]], &v)
			q[f] = append(q[f], v)
		}
		var po [2]float64
		var side float64
		var path [2]float64
		json.Unmarshal(row["pelvis_offset"], &po)
		json.Unmarshal(row["pelvis_side"], &side)
		json.Unmarshal(row["path"], &path)
		off[f] = [3]float64{side + path[0], po[0] + path[1], po[1]}
	}
	travel := math.Hypot(off[c.Frames-1][0]-off[0][0], off[c.Frames-1][1]-off[0][1])
	nodes := make([]string, len(rigNodes))
	for i, jn := range rigNodes {
		nodes[i] = jn[1]
	}
	meta := map[string]any{"clip": map[string]any{"fps": c.FPS, "frames": c.Frames, "loop": c.Loop, "joints": nodes,
		"q": q, "off": off, "marks": c.Marks, "contact": c.Contact}}
	if glb, err := os.ReadFile(a.rigPath(r.Freak, r.Form)); err == nil {
		meta["rig"] = map[string]any{"data": base64.StdEncoding.EncodeToString(glb), "name": r.Freak + " " + r.Form}
	}
	meta["preview"] = a.previewURL(r.ID)
	if ob, ok := r.Take["obstacle"].(map[string]any); ok && c.Scale > 0 { // the vault's box, in studs from the start
		n := func(k string) float64 { f, _ := ob[k].(float64); return f * c.Scale }
		meta["props"] = []map[string]any{{"kind": "box", "forward": n("near_z"), "depth": n("depth"), "height": n("height"), "width": 7.0}}
	}

	settings := [][2]string{{"Freak", r.Freak + " " + r.Form}, {"Length", fmt.Sprintf("%d frames · %.2f s", c.Frames, float64(c.Frames)/30)},
		{"Travel", fmt.Sprintf("%.1f studs", travel)}, {"Style", r.Preset + " · travel " + r.Travel}}
	if e, ok := r.Take["engine"].(string); ok {
		settings = append(settings, [2]string{"Engine", e})
	}
	if g, ok := r.Take["gen_ms"].([]any); ok && len(g) > 0 {
		var parts []string
		for _, x := range g {
			parts = append(parts, fmt.Sprintf("%.1f", x))
		}
		settings = append(settings, [2]string{"Generation", strings.Join(parts, " + ") + " ms"})
	} else if s, ok := r.Take["gen_s"].(float64); ok {
		settings = append(settings, [2]string{"Generation", fmt.Sprintf("%.1f s", s)})
	}
	keys := make([]string, 0, len(r.Params))
	for k := range r.Params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if k == "name" || k == "freak" || k == "form" {
			continue
		}
		v, _ := json.Marshal(r.Params[k])
		s := string(v)
		if s == "null" || s == `""` || s == "0" || s == "[]" {
			continue
		}
		if len(s) > 160 {
			s = s[:157] + "…"
		}
		settings = append(settings, [2]string{k, s})
	}
	if len(r.Marks) > 0 {
		mk, _ := json.Marshal(r.Marks)
		settings = append(settings, [2]string{"Marks (s)", string(mk)})
	}
	saved := []string{a.clipPath(r)}
	saved = append(saved, r.Exported...)
	card := map[string]any{"kind": "clip", "title": r.Name, "status": "done", "clip_id": r.ID, "tool": r.Tool,
		"summary": fmt.Sprintf("%s · %d frames · %.1f studs", r.Tool, c.Frames, travel), "settings": settings, "saved": saved,
		"next": []map[string]string{
			{"label": "Export JSON", "prompt": fmt.Sprintf("Export motion clip %s with motion_export.", r.ID)},
			{"label": "Variation", "prompt": fmt.Sprintf("Make a variation of motion clip %s (same tool and parameters, another seed or a small change).", r.ID)},
		}}
	text := fmt.Sprintf("Clip %s: %s on %s %s, %d frames (%.2f s), travel %.1f studs.\nPreview: %s\nClip JSON: %s",
		r.ID, r.Tool, r.Freak, r.Form, c.Frames, float64(c.Frames)/30, travel, a.previewURL(r.ID), a.clipPath(r))
	return reply{Text: text, Card: card, Meta: meta}, nil
}
