package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// app holds the configuration and the two Python workers. Every job (a worker op or a Blender run) holds jobMu: one
// GPU-heavy job at a time, and our own Blender never runs while a guard looks for someone else's.
type app struct {
	data, here                 string // the data folder; the folder of the exe (python/, blender/ next to it)
	mbRepo, kimodoRepo, hfHome string
	blender, pipeline, game    string
	exportDir, port, skeleton  string
	jobMu                      sync.Mutex
	mb, kimodo                 *worker
}

func env(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func newApp() (*app, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, err
	}
	local, _ := os.UserCacheDir() // %LOCALAPPDATA% on Windows
	a := &app{
		data:       env("MBS_DATA", filepath.Join(local, "MotionBricksStudio")),
		here:       env("MBS_HOME", filepath.Dir(exe)),
		mbRepo:     env("MBS_MOTIONBRICKS", `H:\tools\motionbricks`),
		kimodoRepo: env("MBS_KIMODO", `H:\tools\kimodo`),
		hfHome:     env("MBS_HF_HOME", `H:\tools\hf`),
		blender:    env("MBS_BLENDER", `H:\Blender\blender.exe`),
		pipeline:   env("MBS_PIPELINE", `D:\Pictures\build a freak\_pipeline`),
		game:       env("MBS_GAME", `G:\projects\buildafreak`),
		exportDir:  env("MBS_EXPORT_DIR", `D:\Pictures\build a freak\forms\skull_anim\generated`),
		port:       env("MBS_PORT", "5191"),
	}
	if err := os.MkdirAll(filepath.Join(a.data, "takes"), 0o755); err != nil {
		return nil, err
	}
	a.skeleton = filepath.Join(a.data, "soma_skeleton.npz")
	skel := "MBS_SKELETON=" + a.skeleton
	a.mb = &worker{name: "motionbricks", python: filepath.Join(a.mbRepo, ".venv", "Scripts", "python.exe"),
		script: filepath.Join(a.here, "python", "mb_worker.py"), dir: a.mbRepo, log: filepath.Join(a.data, "motionbricks.log"),
		env: []string{skel, "MBS_MOTIONBRICKS=" + a.mbRepo, "MBS_WARP_CACHE=" + env("MBS_WARP_CACHE", `H:\tools\warp-cache`)}}
	// Kimodo runs text-free: offline, so nothing (Llama, LLM2Vec) can ever be fetched
	a.kimodo = &worker{name: "kimodo", python: filepath.Join(a.kimodoRepo, ".venv", "Scripts", "python.exe"),
		script: filepath.Join(a.here, "python", "kimodo_worker.py"), dir: a.kimodoRepo, log: filepath.Join(a.data, "kimodo.log"),
		env: []string{skel, "HF_HOME=" + a.hfHome, "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1"}}
	return a, nil
}

func (a *app) close() {
	a.mb.stop()
	a.kimodo.stop()
}

// ---- the GPU guard ----

// busyGPU names what is open that the owner keeps the GPU free for: Roblox Studio (its Play crashed the driver with a
// Blender render on 2026-10-26), the Roblox player (a game rendering on the same GPU) and any Blender. Ours never runs
// here: jobMu is held.
func busyGPU() []string {
	out, err := exec.Command("tasklist", "/FO", "CSV", "/NH").Output()
	if err != nil {
		return nil
	}
	rows, _ := csv.NewReader(strings.NewReader(string(out))).ReadAll()
	seen := map[string]int{}
	for _, r := range rows {
		if len(r) > 1 {
			n := strings.ToLower(r[0])
			if strings.HasPrefix(n, "robloxstudio") || strings.HasPrefix(n, "robloxplayer") || n == "blender.exe" {
				seen[r[0]]++
			}
		}
	}
	var busy []string
	for n, c := range seen {
		busy = append(busy, fmt.Sprintf("%s ×%d", n, c))
	}
	return busy
}

// gpuJob runs one worker op under the job lock, after the guard.
func (a *app) gpuJob(ctx context.Context, w *worker, gpuOK bool, op string, args any) (map[string]any, error) {
	a.jobMu.Lock()
	defer a.jobMu.Unlock()
	if !gpuOK {
		if busy := busyGPU(); len(busy) > 0 {
			return nil, fmt.Errorf("the GPU may be in use: %s is running. Ask the owner whether Roblox is out of Play and no Blender render runs, then call again with gpu_ok=true", strings.Join(busy, ", "))
		}
	}
	if _, err := os.Stat(a.skeleton); err != nil && w == a.kimodo {
		if _, err := a.mb.call(ctx, "poses", map[string]any{}); err != nil { // the MotionBricks worker writes it
			return nil, fmt.Errorf("the SOMA skeleton is needed first: %w", err)
		}
	}
	return w.call(ctx, op, args)
}

// ---- workers ----

// worker is one Python process speaking JSON lines (python/proto.py). It starts on first use and stops after
// idleStop without a job, so the models leave the GPU when nobody is generating.
type worker struct {
	name, python, script, dir, log string
	env                            []string
	mu                             sync.Mutex
	cmd                            *exec.Cmd
	in                             *bufio.Writer
	lines                          chan string
	next                           int
	idle                           *time.Timer
	Ready                          map[string]any
}

const idleStop = 10 * time.Minute

func (w *worker) start() error {
	if _, err := os.Stat(w.python); err != nil {
		return fmt.Errorf("%s runtime not found: %s", w.name, w.python)
	}
	logf, err := os.OpenFile(w.log, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	cmd := exec.Command(w.python, "-u", w.script)
	cmd.Dir = w.dir
	cmd.Env = append(os.Environ(), w.env...)
	cmd.Stderr = logf
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	if err := cmd.Start(); err != nil {
		logf.Close()
		return err
	}
	w.cmd, w.in, w.lines = cmd, bufio.NewWriter(stdin), make(chan string, 4)
	go func() {
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 1<<20), 64<<20)
		for sc.Scan() {
			w.lines <- sc.Text()
		}
		close(w.lines)
		cmd.Wait()
		logf.Close()
	}()
	// the first line says the models are loaded (MotionBricks ~15 s, Kimodo longer on a cold cache)
	r, err := w.read(context.Background(), 0, 10*time.Minute)
	if err != nil {
		w.kill()
		return err
	}
	w.Ready = r
	return nil
}

func (w *worker) read(ctx context.Context, id int, limit time.Duration) (map[string]any, error) {
	timeout := time.After(limit)
	for {
		select {
		case line, ok := <-w.lines:
			if !ok {
				w.cmd = nil
				return nil, fmt.Errorf("the %s worker exited; its log: %s\n%s", w.name, w.log, tail(w.log, 12))
			}
			var rep struct {
				ID     int            `json:"id"`
				OK     bool           `json:"ok"`
				Result map[string]any `json:"result"`
				Error  string         `json:"error"`
			}
			if json.Unmarshal([]byte(line), &rep) != nil || rep.ID != id {
				continue
			}
			if !rep.OK {
				return nil, fmt.Errorf("%s: %s", w.name, rep.Error)
			}
			return rep.Result, nil
		case <-timeout:
			w.kill()
			return nil, fmt.Errorf("the %s worker did not answer within %s; stopped it", w.name, limit)
		case <-ctx.Done():
			w.kill() // a half-done job leaves the worker in an unknown state
			return nil, ctx.Err()
		}
	}
}

func (w *worker) call(ctx context.Context, op string, args any) (map[string]any, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.idle != nil {
		w.idle.Stop()
	}
	if w.cmd == nil {
		if err := w.start(); err != nil {
			return nil, err
		}
	}
	w.next++
	b, _ := json.Marshal(map[string]any{"id": w.next, "op": op, "args": args})
	w.in.Write(append(b, '\n'))
	if err := w.in.Flush(); err != nil {
		w.kill()
		return nil, fmt.Errorf("the %s worker is gone: %v", w.name, err)
	}
	r, err := w.read(ctx, w.next, 5*time.Minute)
	if w.cmd != nil {
		w.idle = time.AfterFunc(idleStop, w.stop)
	}
	return r, err
}

// kill stops our own process (by its handle, so by our PID only).
func (w *worker) kill() {
	if w.cmd != nil && w.cmd.Process != nil {
		w.cmd.Process.Kill()
	}
	w.cmd = nil
}

func (w *worker) stop() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.kill()
}

func tail(path string, n int) string {
	b, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimRight(string(b), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// blocking runs python/blocking.py once with the MotionBricks venv's python: a recipe's keys interpolated, no model,
// no GPU. op is the worker op the recipe stands behind.
func (a *app) blocking(ctx context.Context, op string, args map[string]any) (map[string]any, error) {
	if _, err := os.Stat(a.skeleton); err != nil {
		return nil, errors.New("the SOMA skeleton is not there yet: run any MotionBricks tool once (motion_poses)")
	}
	recipe := map[string]string{"generate": "keyed"}[op]
	if recipe == "" {
		recipe = op
	}
	args = maps.Clone(args)
	delete(args, "steps")
	b, _ := json.Marshal(args)
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, a.mb.python, filepath.Join(a.here, "python", "blocking.py"), recipe, string(b))
	cmd.Env = append(os.Environ(), "MBS_SKELETON="+a.skeleton)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		e := strings.TrimSpace(stderr.String())
		if i := strings.LastIndex(e, "\n"); i >= 0 {
			e = e[i+1:]
		}
		return nil, fmt.Errorf("blocking preview failed: %v: %s", err, e)
	}
	var meta map[string]any
	return meta, json.Unmarshal(out, &meta)
}

// ---- Blender ----

// retarget puts a take on a freak with blender/retarget.py; it writes the clip JSON (and the rig preview GLB once).
func (a *app) retarget(ctx context.Context, job map[string]any) error {
	a.jobMu.Lock()
	defer a.jobMu.Unlock()
	job["pipeline"], job["game"] = a.pipeline, a.game
	jf := filepath.Join(job["take_dir"].(string), "retarget_job.json")
	b, _ := json.MarshalIndent(job, "", "  ")
	if err := os.WriteFile(jf, b, 0o644); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, a.blender, "-b", "--factory-startup", "--python", filepath.Join(a.here, "blender", "retarget.py"), "--", jf)
	out, err := cmd.CombinedOutput()
	logPath := filepath.Join(job["take_dir"].(string), "retarget.log")
	os.WriteFile(logPath, out, 0o644)
	if strings.Contains(string(out), "RETARGET-OK") {
		return nil
	}
	if i := strings.Index(string(out), "RETARGET-ERROR"); i >= 0 {
		return errors.New(strings.SplitN(string(out[i+15:]), "\n", 2)[0])
	}
	if err == nil {
		err = errors.New("no clip written")
	}
	return fmt.Errorf("Blender retarget failed (%v); log %s:\n%s", err, logPath, tail(logPath, 15))
}
