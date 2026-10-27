// node dev/serve.mjs [port]: serves the card test host and drives the real
// server (../motionbricks-studio.exe) over MCP stdio for it.
import http from "node:http";
import { spawn } from "node:child_process";
import { readFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const port = Number(process.argv[2] || 5181);
const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const dl = join(tmpdir(), "mbs-card-host");
mkdirSync(dl, { recursive: true });

const srv = spawn(here("../../motionbricks-studio.exe"), [], { stdio: ["pipe", "pipe", "inherit"] });
let buf = "";
let next = 1;
const waiting = new Map();
srv.stdout.on("data", (d) => {
  buf += d;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const m = JSON.parse(buf.slice(0, i));
    buf = buf.slice(i + 1);
    waiting.get(m.id)?.(m);
    waiting.delete(m.id);
  }
});
const rpc = (method, params) =>
  new Promise((res) => {
    const id = next++;
    waiting.set(id, res);
    srv.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
  });

await rpc("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "card-host", version: "0" } });
const tools = (await rpc("tools/list", {})).result.tools;
const uri = tools.find((t) => t._meta)?._meta.ui.resourceUri;
const resource = (await rpc("resources/read", { uri })).result.contents[0];
const hostJS = (await build({ entryPoints: [here("./host.js")], bundle: true, format: "iife", write: false })).outputFiles[0].text;

http
  .createServer(async (req, res) => {
    const path = new URL(req.url, "http://host").pathname;
    const send = (type, body) => {
      res.writeHead(200, { "content-type": type });
      res.end(body);
    };
    if (path === "/") return send("text/html", readFileSync(here("./host.html")));
    if (path === "/host.js") return send("text/javascript", hostJS);
    if (path === "/card.html") return send("text/html", resource.text);
    if (path === "/meta") return send("application/json", JSON.stringify(resource._meta.ui));
    if (path === "/dl") return send("text/plain", dl.replace(/\\/g, "/"));
    if (path === "/call") {
      let body = "";
      for await (const c of req) body += c;
      const r = await rpc("tools/call", JSON.parse(body));
      return send("application/json", JSON.stringify(r.result ?? { isError: true, content: [{ type: "text", text: JSON.stringify(r.error) }] }));
    }
    res.writeHead(404).end();
  })
  .listen(port, () => console.log(`card test host on http://localhost:${port}`));
