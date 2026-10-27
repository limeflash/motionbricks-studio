"""The workers' JSON-lines protocol with the Go server: one request per stdin line {"id", "op", "args"}, one reply per
line {"id", "ok", "result" | "error"}. Anything else a library prints (warp, torch, TensorRT) goes to stderr: fd 1 is
pointed at fd 2 and the replies go to a private copy of the original stdout."""
import json
import os
import sys
import time
import traceback

_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def send(obj):
    _out.write(json.dumps(obj) + "\n")
    _out.flush()


def serve(ops, ready):
    send({"id": 0, "ok": True, "result": ready})
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        t = time.perf_counter()
        try:
            res = ops[req["op"]](**(req.get("args") or {}))
            res = dict(res or {}, wall_s=round(time.perf_counter() - t, 3))
            send({"id": req["id"], "ok": True, "result": res})
        except Exception as e:                      # the server shows it; the worker stays up for the next job
            traceback.print_exc()
            send({"id": req["id"], "ok": False, "error": f"{type(e).__name__}: {e}"})
