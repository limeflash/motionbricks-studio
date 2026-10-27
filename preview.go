package main

import (
	"encoding/json"
	"log"
	"net"
	"net/http"
	"strings"
)

// The local preview server: the card page for any clip at http://127.0.0.1:<port>/clip/<id>, fed by /api/<id> (the
// same result the tool returns). For the desktop browser; the chat card carries its data inline and needs no server.
func (a *app) previewURL(id string) string { return "http://127.0.0.1:" + a.port + "/clip/" + id }

func (a *app) servePreview() {
	ln, err := net.Listen("tcp", "127.0.0.1:"+a.port)
	if err != nil {
		log.Printf("preview server not started (%v): another instance may be serving it", err)
		return
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if rs := a.recent(1); r.URL.Path == "/" && len(rs) > 0 {
			http.Redirect(w, r, "/clip/"+rs[0].ID, http.StatusFound)
			return
		}
		http.NotFound(w, r)
	})
	mux.HandleFunc("/clip/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/clip/")
		page := strings.Replace(cardHTML, "<head>", `<head><script>window.MBS_STANDALONE = `+jsonString("/api/"+id)+`;</script>`, 1)
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(page))
	})
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
	http.Serve(ln, mux)
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
