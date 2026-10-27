// MotionBricks Studio clip card (an MCP App): the generated clip playing on the freak, the parameters used, the
// saved paths and next steps. Everything arrives in the tool result (structuredContent = the card, _meta.motion = the
// rig GLB and the clip's frames); it never calls back into the server. Opened from the local preview server
// (window.MBS_STANDALONE = its /api URL) it fetches the same result instead.
import { App, applyHostFonts } from "@modelcontextprotocol/ext-apps/app-with-deps";
import { Player } from "./viewer.js";

const ICON = {
  logo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="13" cy="4.5" r="2"/><path d="M9 21l2.5-6 3 2.5V22M7 11l3.5-3.5 4 1.5 2.5 3.5M10.5 7.5 11 14l-4 2"/></svg>',
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l12-7.5z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4.5" width="4" height="15" rx="1"/><rect x="14" y="4.5" width="4" height="15" rx="1"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2.5"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.5h.01"/></svg>',
  send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
};
const PENDING = {
  motion_hit_reaction: "Generating the hit reaction",
  motion_transition: "Generating the transition",
  motion_from_keys: "Kimodo: motion through the keys",
  motion_vault: "Kimodo: the vault",
  motion_locomotion: "Kimodo: walking the path",
  motion_retarget: "Retargeting in Blender",
  motion_export: "Exporting",
  motion_show: "Loading",
};

const root = document.getElementById("root");
const standalone = window.MBS_STANDALONE;
const app = standalone ? null : new App({ name: "MotionBricks Studio card", version: __VERSION__ }, {}, { autoResize: true });
const state = { input: null, result: null, cancelled: null, started: Date.now() };
let player = null;
let loadedKey = "";

function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v; // only our own ICON constants
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : String(kid));
  return el;
}

function toast(text) {
  const t = document.getElementById("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => t.classList.remove("show"), 2200);
}
async function copy(text, what = "Copied") {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = h("textarea", { style: "position:fixed;opacity:0" }, text);
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  toast(what);
}
async function ask(prompt) {
  if (app?.getHostCapabilities()?.message) {
    try {
      await app.sendMessage({ role: "user", content: [{ type: "text", text: prompt }] });
      toast("Sent to the chat");
      return;
    } catch {}
  }
  copy(prompt, "Copied: paste it in the chat");
}

function header(title, sub, status) {
  const [label, cls] = { done: ["Ready", "ok"], warn: ["Check", "quote"], running: ["Working", "run"], failed: ["Failed", "bad"] }[status] || [];
  return h("div", { class: "head" },
    h("div", { class: "mark", html: ICON.logo, "aria-hidden": "true" }),
    h("div", { class: "titles" }, h("div", { class: "title" }, title || "MotionBricks Studio"), sub && h("div", { class: "sub" }, sub)),
    label && h("span", { class: `pill ${cls}` }, h("i"), label));
}

function alertBlock(text) {
  return h("div", { class: "alert", role: "alert" }, h("span", { html: ICON.alert, "aria-hidden": "true" }), h("div", {}, text));
}

function stage(meta) {
  const el = h("div", { class: "stage loading" });
  const center = h("div", { class: "center" }, h("div", {}, h("div", { class: "spinner" }), "Loading the freak…"));
  el.append(center);
  if (!meta.rig) {
    center.replaceChildren(h("div", {}, h("b", {}, "No preview model"), "The rig GLB is built on the first retarget."));
    return el;
  }
  const clip = meta.clip;
  if (player) player.attach(el);
  else player = window.mbsPlayer = new Player(el);
  const p = player;
  const key = `${meta.rig.data.length}:${clip.frames}:${clip.q[0][0].join(",")}:${clip.q.at(-1)[3].join(",")}`;
  const ready = () => {
    center.remove();
    el.classList.remove("loading");
    el.classList.add("ready");
    el.append(controls(p, clip));
  };
  if (loadedKey === key && p.clip) {
    ready();
    return el;
  }
  loadedKey = key;
  p.load(meta.rig, clip).then(() => {
    p.setProps(meta.props);
    p.setKeys(clip.keys || []);
    ready();
    app?.sendLog?.({ level: "info", data: "clip preview loaded" }).catch(() => {});
  }, (err) => {
    app?.sendLog?.({ level: "error", data: `clip preview failed: ${err?.message}` }).catch(() => {});
    el.classList.remove("loading");
    center.replaceChildren(h("div", {}, h("b", {}, "Preview unavailable"), String(err?.message || err).slice(0, 200)));
  });
  return el;
}

function controls(p, clip) {
  const playBtn = h("button", { class: "icon-btn", "aria-label": "Play / pause", html: p.playing ? ICON.pause : ICON.play });
  playBtn.onclick = () => {
    p.playing = !p.playing;
    if (p.playing) p.focusKey(-1);
    playBtn.innerHTML = p.playing ? ICON.pause : ICON.play;
  };
  const range = h("input", { type: "range", min: 0, max: clip.frames - 1, value: p.frame || 0, "aria-label": "Frame" });
  range.oninput = () => {
    p.playing = false;
    playBtn.innerHTML = ICON.play;
    p.seek(Number(range.value));
  };
  const fr = h("span", { class: "fr" });
  const R = h("i", {}, "R");
  const L = h("i", {}, "L");
  const marks = Object.entries(clip.marks || {});
  const keys = clip.keys || [];
  p.onframe = (f) => {
    range.value = f;
    const k = keys.indexOf(f);
    const tag = k >= 0 ? `K${k + 1}` : marks.find(([, s]) => Math.abs(s * clip.fps - f) < 0.5)?.[0];
    fr.textContent = `${f + 1}/${clip.frames}${tag ? " " + tag : ""}`;
    const c = clip.contact?.[f] || 0;
    R.className = c & 1 ? "on" : "";
    L.className = c & 2 ? "on" : "";
  };
  p.onframe(p.frame || 0);
  const seg = (items, cur, set) => {
    const g = h("div", { class: "seg" });
    for (const [v, label] of items) {
      const b = h("button", { "aria-pressed": String(v === cur()) }, label);
      b.onclick = () => {
        set(v);
        for (const x of g.children) x.setAttribute("aria-pressed", String(x === b));
      };
      g.append(b);
    }
    return g;
  };
  let view = "side";
  const keysBtn = keys.length > 0 && h("button", { "aria-pressed": String(p.keyGroup.visible), title: "The key poses along the path" }, "Keys");
  if (keysBtn) keysBtn.onclick = () => keysBtn.setAttribute("aria-pressed", String((p.keyGroup.visible = !p.keyGroup.visible)));
  return h("div", { class: "bar glass" }, playBtn, range, fr, h("span", { class: "feet", title: "Planted feet" }, L, R),
    keysBtn && h("div", { class: "seg" }, keysBtn),
    seg([[0.25, "¼×"], [0.5, "½×"], [1, "1×"]], () => p.speed, (v) => (p.speed = v)),
    seg([["side", "Side"], ["3/4", "¾"], ["front", "Front"]], () => view, (v) => p.view((view = v))));
}

// the key poses: one button each (its frame, and how far the model's motion came out from it), so every pose the
// caller set can be looked at on the freak
function keysBlock(meta) {
  const clip = meta.clip || {};
  if (!clip.keys?.length || !player) return null;
  const miss = clip.key_miss_cm || [];
  return h("div", {}, h("div", { class: "label" }, "Key poses", h("em", {}, "tap to look at a pose on the freak")),
    h("div", { class: "chips" }, ...clip.keys.map((f, i) => {
      const m = miss[i];
      return h("button", {
        class: `chip${m > 10 ? " bad" : " on"}`, title: m != null ? `missed by ${m} cm` : `frame ${f + 1}`,
        onclick: () => { player.playing = false; player.focusKey(i); player.seek(Math.min(f, clip.frames - 1)); },
      }, h("i"), `K${i + 1} · ${f + 1}` + (m != null ? ` · ${Math.round(m)} cm` : ""));
    })));
}

function warningsBlock(ws) {
  if (!ws?.length) return null;
  return h("div", { class: "warns", role: "status" }, h("b", {}, "Something looks wrong"),
    h("ul", {}, ...ws.map((w) => h("li", {}, w))));
}

function kvs(settings) {
  return h("div", {}, h("div", { class: "label" }, "Parameters"),
    h("div", { class: "kvs" }, ...settings.map(([k, v]) => h("div", { class: "kv", title: v }, h("small", {}, k), h("b", {}, v)))));
}

function facts(d, meta) {
  const rows = [["Clip", d.clip_id], ...(d.saved || []).map((s, i) => [i === 0 ? "JSON" : "Export", s])];
  if (meta.preview) rows.push(["Preview", meta.preview]);
  return h("div", { class: "facts" }, ...rows.map(([k, v]) =>
    h("div", { class: "fact" }, h("small", {}, k), h("code", { title: v }, v),
      h("button", { class: "ghost", onclick: () => copy(v), "aria-label": "Copy " + k, html: ICON.copy }))));
}

function nextBlock(next) {
  if (!next?.length || standalone) return null;
  return h("div", { class: "actions" }, ...next.map((a, i) =>
    h("button", { class: `btn${i === 0 ? " primary" : ""}`, onclick: () => ask(a.prompt) }, a.label, h("span", { html: ICON.send, "aria-hidden": "true" }))));
}

function footer() {
  return h("div", { class: "foot" }, h("span", {}, "NVIDIA MotionBricks + Kimodo, offline · our own poses only"), h("span", {}, `v${__VERSION__}`));
}

function render() {
  const r = state.result;
  let view;
  if (!r) {
    const tool = app?.getHostContext()?.toolInfo?.tool?.name || "";
    view = [header(PENDING[tool] || "Working", state.cancelled ? `Cancelled: ${state.cancelled}` : "", state.cancelled ? "failed" : "running"),
      h("div", { class: "stage" }, h("div", { class: "center" }, h("div", {}, h("div", { class: "spinner" }), h("b", {}, PENDING[tool] || "Working"), "MotionBricks takes a second, Kimodo a few, the Blender retarget about ten."))),
      footer()];
  } else {
    const d = r.structuredContent;
    const meta = r._meta?.motion || {};
    const text = (r.content || []).filter((c) => c.type === "text").map((c) => c.text).join("\n");
    if (!d || d.kind === "error") view = [header(d?.title || "MotionBricks Studio", "", "failed"), alertBlock(d?.error || text), footer()];
    else view = [header(d.title, d.summary, r.isError ? "failed" : d.status), r.isError ? alertBlock(d.error) : null,
      meta.clip ? stage(meta) : null, warningsBlock(d.warnings), meta.clip ? keysBlock(meta) : null, kvs(d.settings || []),
      facts(d, meta), nextBlock(d.next), footer()];
  }
  root.replaceChildren(...view.filter(Boolean));
}

if (standalone) {
  fetch(standalone).then((x) => x.json()).then((r) => {
    state.result = r;
    render();
  });
  render();
} else {
  app.onhostcontextchanged = (ctx) => ctx?.styles?.css?.fonts && applyHostFonts(ctx.styles.css.fonts);
  app.ontoolinput = (p) => {
    state.input = p.arguments || {};
    render();
  };
  app.ontoolresult = (r) => {
    state.result = r;
    render();
  };
  app.ontoolcancelled = (p) => {
    state.cancelled = p?.reason || "the tool call was cancelled";
    render();
  };
  render();
  app.connect().then(() => {
    const f = app.getHostContext()?.styles?.css?.fonts;
    if (f) applyHostFonts(f);
    render();
  });
}
