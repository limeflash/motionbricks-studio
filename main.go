// motionbricks-studio is a stdio MCP server that makes character motion offline with NVIDIA MotionBricks and Kimodo
// (Python workers on H:), puts it on a Build a Freak freak through the art pipeline's Blender retarget, and shows each
// clip in a widget card and on a local preview page. Its shape follows tripo-studio-mcp.
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"sync"
)

const (
	serverName    = "motionbricks-studio"
	serverVersion = "0.1.0"
)

const instructions = `Offline character motion for Build a Freak: NVIDIA MotionBricks (transitions, hit reactions; ~4 ms a generation) and Kimodo (motion through key poses, end-effector targets and a root path; ~2-3 s), retargeted onto a freak (SKULL heavy for now) and exported as the game's clip JSON.
You are the text layer: Kimodo runs text-free. Turn a description into constraints: key poses (pose specs on SOMA, see motion_poses), frames, hand/foot targets and the root's path.
For reactions, knock-backs, vaults, a limp, transitions and idle drafts. NOT for strikes: attacks stay hand-keyed in Blender (the owner rejected generated punches). Keep motion smooth: no snapped time-warps.
Only our own poses: never BONES sample motions or poses taken from them.
One GPU job at a time, and none while Roblox is in Play or a Blender render runs: tools refuse while Roblox (Studio or the player) or a Blender is open; ask the owner, then pass gpu_ok=true.
Every generation returns a clip id and a card that plays the clip on the freak; motion_export writes the clip JSON for art/freaks/skull/scripts/clips_export.py. Keep your own reply short: the card shows the parameters.`

// The result widget (MCP Apps). ui/build.mjs writes it; go generate rebuilds it.
//
//go:generate npm --prefix ui run build
//go:embed ui/dist/card.html
var cardHTML string

const appMIME = "text/html;profile=mcp-app"

type message struct {
	ID     json.RawMessage `json:"id,omitempty"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params,omitempty"`
}

type server struct {
	app     *app
	cardURI string
	mu      sync.Mutex // one JSON line at a time on stdout
	cancels sync.Map   // request id -> context.CancelFunc
	running sync.WaitGroup
}

func sha256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func main() {
	log.SetFlags(0)
	log.SetPrefix(serverName + ": ")
	a, err := newApp()
	if err != nil {
		log.Fatal(err)
	}
	defer a.close()
	go a.servePreview()
	// The URI changes with each widget build, so a host never keeps a stale copy.
	s := &server{app: a, cardURI: fmt.Sprintf("ui://motionbricks-studio/card-%s.html", sha256Hex([]byte(cardHTML))[:10])}
	in := bufio.NewReader(os.Stdin)
	for {
		line, err := in.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			var m message
			if json.Unmarshal(line, &m) != nil {
				s.send(map[string]any{"jsonrpc": "2.0", "id": nil, "error": map[string]any{"code": -32700, "message": "parse error"}})
			} else {
				s.running.Add(1)
				go func() { defer s.running.Done(); s.handle(m) }()
			}
		}
		if err != nil {
			s.running.Wait() // stdin closed: finish what is in flight, then exit (the deferred close stops the workers)
			return
		}
	}
}

func (s *server) handle(m message) {
	switch m.Method {
	case "":
	case "initialize":
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		_ = json.Unmarshal(m.Params, &p)
		if p.ProtocolVersion == "" {
			p.ProtocolVersion = "2025-06-18"
		}
		s.result(m.ID, map[string]any{
			"protocolVersion": p.ProtocolVersion,
			"capabilities":    map[string]any{"tools": map[string]any{}, "resources": map[string]any{}},
			"serverInfo":      map[string]any{"name": serverName, "version": serverVersion},
			"instructions":    instructions,
		})
	case "ping":
		s.result(m.ID, map[string]any{})
	case "tools/list":
		s.result(m.ID, map[string]any{"tools": s.toolList()})
	case "tools/call":
		s.call(m)
	case "resources/list":
		s.result(m.ID, map[string]any{"resources": []map[string]any{{
			"uri": s.cardURI, "name": "Motion clip card", "mimeType": appMIME,
			"description": "A generated clip playing on the freak, with the parameters used.",
		}}})
	case "resources/read":
		var p struct {
			URI string `json:"uri"`
		}
		if json.Unmarshal(m.Params, &p) != nil || p.URI != s.cardURI {
			s.fail(m.ID, -32002, "resource not found: "+p.URI)
			return
		}
		s.result(m.ID, map[string]any{"contents": []map[string]any{{
			"uri": s.cardURI, "mimeType": appMIME, "text": cardHTML,
			"_meta": map[string]any{"ui": map[string]any{"prefersBorder": true,
				"csp": map[string]any{"connectDomains": []string{"data:", "blob:"}, "resourceDomains": []string{"data:", "blob:"}}}},
		}}})
	case "notifications/cancelled":
		var p struct {
			RequestID json.RawMessage `json:"requestId"`
		}
		if json.Unmarshal(m.Params, &p) == nil {
			if cancel, ok := s.cancels.Load(string(p.RequestID)); ok {
				cancel.(context.CancelFunc)()
			}
		}
	default:
		if len(m.ID) > 0 {
			s.fail(m.ID, -32601, "method not found: "+m.Method)
		}
	}
}

func (s *server) toolList() []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		m := map[string]any{"name": t.Name, "description": t.Description, "inputSchema": t.InputSchema}
		if t.UI {
			m["_meta"] = map[string]any{
				"ui":             map[string]any{"resourceUri": s.cardURI, "visibility": []string{"model", "app"}},
				"ui/resourceUri": s.cardURI, // the pre-GA key some hosts still read
			}
		}
		out = append(out, m)
	}
	return out
}

func (s *server) call(m message) {
	var p struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(m.Params, &p); err != nil {
		s.fail(m.ID, -32602, "bad tools/call params")
		return
	}
	var t *tool
	for i := range tools {
		if tools[i].Name == p.Name {
			t = &tools[i]
		}
	}
	if t == nil {
		s.fail(m.ID, -32602, "unknown tool: "+p.Name)
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	key := string(m.ID)
	s.cancels.Store(key, cancel)
	defer func() { s.cancels.Delete(key); cancel() }()

	r, err := func() (r reply, err error) {
		defer func() {
			if v := recover(); v != nil {
				err = fmt.Errorf("internal error: %v", v)
			}
		}()
		return t.run(ctx, s.app, p.Arguments)
	}()
	if err != nil && strings.HasPrefix(err.Error(), "bad arguments") {
		var names []string
		if ps, ok := t.InputSchema["properties"].(map[string]any); ok {
			for k := range ps {
				names = append(names, k)
			}
		}
		sort.Strings(names)
		err = fmt.Errorf("%w. %s takes: %s", err, t.Name, strings.Join(names, ", "))
	}
	s.result(m.ID, r.result(t, err))
}

func (s *server) result(id json.RawMessage, result any) {
	s.send(map[string]any{"jsonrpc": "2.0", "id": id, "result": result})
}

func (s *server) fail(id json.RawMessage, code int, msg string) {
	s.send(map[string]any{"jsonrpc": "2.0", "id": id, "error": map[string]any{"code": code, "message": msg}})
}

func (s *server) send(v any) {
	b, err := json.Marshal(v)
	if err != nil {
		log.Printf("encode: %v", err)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	os.Stdout.Write(append(b, '\n'))
}
