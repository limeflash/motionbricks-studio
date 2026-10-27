package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// The local preview server: the card page for any clip at http://127.0.0.1:<port>/clip/<id>, fed by /api/<id> (the
// same result the tool returns), and the pose editor at /edit/<id> for clips made from key poses. For the desktop
// browser; the chat card carries its data inline and needs no server.
func (a *app) previewURL(id string) string { return "http://127.0.0.1:" + a.port + "/clip/" + id }
func (a *app) editURL(id string) string    { return "http://127.0.0.1:" + a.port + "/edit/" + id }
func (a *app) keysPath(id string) string   { return filepath.Join(a.dir(id), "keys.json") }

func (a *app) servePreview() {
	ln, err := net.Listen("tcp", "127.0.0.1:"+a.port)
	if err != nil {
		log.Printf("preview server not started (%v): another instance may be serving it", err)
		return
	}
	page := func(html, global, value string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			v := jsonString(value + strings.TrimPrefix(r.URL.Path, r.Pattern))
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write([]byte(strings.Replace(html, "<head>", `<head><script>window.`+global+` = `+v+`;</script>`, 1)))
		}
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if rs := a.recent(1); r.URL.Path == "/" && len(rs) > 0 {
			http.Redirect(w, r, "/clip/"+rs[0].ID, http.StatusFound)
			return
		}
		http.NotFound(w, r)
	})
	mux.HandleFunc("/clip/", page(cardHTML, "MBS_STANDALONE", "/api/"))
	mux.HandleFunc("/edit/", page(editorHTML, "MBS_EDIT", ""))
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		var res map[string]any
		rec, err := a.load(strings.TrimPrefix(r.URL.Path, "/api/"))
		if err == nil {
			var rep reply
			rep, err = a.clipReply(rec)
			res = rep.result(nil, err)
		} else {
			res = reply{}.result(nil, err)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(res)
	})
	mux.HandleFunc("GET /api/keys/{id}", func(w http.ResponseWriter, r *http.Request) {
		if _, err := a.load(r.PathValue("id")); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		http.ServeFile(w, r, a.keysPath(r.PathValue("id")))
	})
	mux.HandleFunc("GET /api/skin", func(w http.ResponseWriter, r *http.Request) {
		path := filepath.Join(a.data, "soma_skin.json")
		if _, err := os.Stat(path); err != nil {
			if err := a.makeSkin(r.Context(), path); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		http.ServeFile(w, r, path)
	})
	// a run from the editor starts Blender and maybe the GPU: only this page may ask (the Host and Origin checks keep
	// other sites and DNS rebinding out)
	mux.HandleFunc("POST /api/edit/{id}", func(w http.ResponseWriter, r *http.Request) {
		self := "127.0.0.1:" + a.port
		if o := r.Header.Get("Origin"); r.Host != self || (o != "" && o != "http://"+self) {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		var p editArgs
		var res map[string]any
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			res = reply{}.result(nil, fmt.Errorf("bad request: %v", err))
		} else {
			rep, err := a.edit(r.Context(), r.PathValue("id"), p)
			res = rep.result(nil, err)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(res)
	})
	http.Serve(ln, mux)
}

type editArgs struct {
	Engine string            `json:"engine"` // blocking | kimodo
	Keys   []json.RawMessage `json:"keys"`   // [{frame, pose: [77][7]}], all of the clip's keys
	Seed   int               `json:"seed"`
	GPUOK  bool              `json:"gpu_ok"`
}

// edit runs a clip again through its key poses as changed in the editor: as blocking (seconds, no GPU) or through
// Kimodo, into a new clip <name>_edit with the same tool, freak, style and marks.
func (a *app) edit(ctx context.Context, id string, p editArgs) (reply, error) {
	rec, err := a.load(id)
	if err != nil {
		return reply{}, err
	}
	b, err := os.ReadFile(a.keysPath(id))
	if err != nil {
		return reply{}, fmt.Errorf("clip %s has no key poses saved (made before the editor, or not from keys)", id)
	}
	var k map[string]any
	if err := json.Unmarshal(b, &k); err != nil {
		return reply{}, err
	}
	if old, _ := k["keys"].([]any); len(old) != len(p.Keys) {
		return reply{}, fmt.Errorf("expected %d keys, got %d", len(old), len(p.Keys))
	}
	k["keys"] = raws(p.Keys)
	file := filepath.Join(a.data, "edits", fmt.Sprintf("%s-%d.json", id, time.Now().UnixNano()))
	os.MkdirAll(filepath.Dir(file), 0o755)
	b, _ = json.Marshal(k)
	if err := os.WriteFile(file, b, 0o644); err != nil {
		return reply{}, err
	}
	defer os.Remove(file)
	c := common{Name: strings.TrimSuffix(strings.TrimSuffix(rec.Name, "_edit"), "_blocking") + "_edit",
		Freak: rec.Freak, Form: rec.Form, GPUOK: p.GPUOK}
	w := a.kimodo
	if p.Engine == "blocking" {
		w = nil
	}
	params := map[string]any{"edited_from": id, "engine": p.Engine, "seed": p.Seed}
	return a.generate(ctx, w, c, rec.Tool, "edited", map[string]any{"file": file, "seed": p.Seed}, params,
		rec.Preset, rec.Travel, func(map[string]any) map[string]any { return rec.Marks })
}

// makeSkin writes the editor's SOMA body from Kimodo's skin (python/skin.py), once.
func (a *app) makeSkin(ctx context.Context, out string) error {
	if _, err := os.Stat(a.skeleton); err != nil {
		return fmt.Errorf("the SOMA skeleton is not there yet: run any MotionBricks tool once (motion_poses)")
	}
	src := filepath.Join(a.kimodoRepo, "kimodo", "assets", "skeletons", "somaskel77", "skin_standard.npz")
	cmd := exec.CommandContext(ctx, a.mb.python, filepath.Join(a.here, "python", "skin.py"), src, out)
	cmd.Env = append(os.Environ(), "MBS_SKELETON="+a.skeleton)
	if b, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("the SOMA body failed: %v: %s", err, strings.TrimSpace(string(b)))
	}
	return nil
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
