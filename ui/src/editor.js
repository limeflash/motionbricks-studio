// The pose editor (http://127.0.0.1:<port>/edit/<clip id>, served by preview.go): a clip's key poses on the SOMA body,
// standing along the hips' path as in MotionBricks' own demo. Pick a key; drag its hands, feet and hips like IK
// controls (the feet stay planted while the body moves) or turn a joint with the rings; then run the keys again as
// blocking or through Kimodo, and the new clip's card opens below. Model space is three.js's own: Y up, the body faces
// +Z, its left is +X, meters. A pose is [77][7] local transforms (px, py, pz, qx, qy, qz, qw), parents first.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const CHAIN = { LeftHand: ["LeftArm", "LeftForeArm"], RightHand: ["RightArm", "RightForeArm"],
  LeftFoot: ["LeftLeg", "LeftShin"], RightFoot: ["RightLeg", "RightShin"] };
const POLE = { LeftHand: [0, -1, 0.3], RightHand: [0, -1, 0.3], LeftFoot: [0, 0, 1], RightFoot: [0, 0, 1] }; // a straight limb's bend
// the joints turned with the rings (white dots; Spine1 sits on the hips and is left to them); the IK ends (orange) and
// the hips (yellow) also move
const TURN = ["Spine2", "Chest", "Neck1", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftLeg", "LeftShin", "LeftToeBase"];
for (const n of [...TURN]) if (n.startsWith("Left")) TURN.push("Right" + n.slice(4));
const COLOR = { start: 0x3ddc84, key: 0x4aa8ff, picked: 0xf2f2f2, also: 0xbfd2ff };

const $ = (s) => document.querySelector(s);
const id = window.MBS_EDIT;
let names, parents, J, poses, origRows, skin, sel = -1, ctl = null, mode = "translate", drag = null;
const picked = new Set(); // the key being edited and those ctrl / shift-clicked with it
const undo = [];

// ---- pose math (the browser twin of python/soma.py's fk, set_world_rot, aim and two_bone) ----
const toPose = (rows) => rows.map((r) => ({ p: new THREE.Vector3(r[0], r[1], r[2]), q: new THREE.Quaternion(r[3], r[4], r[5], r[6]) }));
const toRows = (pose) => pose.map(({ p, q }) => [p.x, p.y, p.z, q.x, q.y, q.z, q.w].map((v) => +v.toFixed(5)));
const clone = (pose) => pose.map(({ p, q }) => ({ p: p.clone(), q: q.clone() }));

function fk(pose) {
  const W = [];
  pose.forEach(({ p, q }, j) => {
    const par = W[parents[j]];
    W[j] = par ? { p: p.clone().applyQuaternion(par.q).add(par.p), q: par.q.clone().multiply(q) } : { p: p.clone(), q: q.clone() };
  });
  return W;
}

function setWorldRot(pose, j, q, W = fk(pose)) {
  const par = W[parents[j]];
  pose[j].q.copy(par ? par.q.clone().invert().multiply(q) : q);
}

function aim(pose, j, child, target) {
  const W = fk(pose);
  const d0 = W[child].p.clone().sub(W[j].p).normalize();
  const d1 = target.clone().sub(W[j].p).normalize();
  setWorldRot(pose, j, new THREE.Quaternion().setFromUnitVectors(d0, d1).multiply(W[j].q), W);
}

// the end on target (clamped to reach), the knee / elbow kept in its current plane; the end keeps its world turn
function twoBone(pose, end, target) {
  const [up, lo] = CHAIN[end].map((n) => J[n]);
  const e = J[end];
  const W = fk(pose);
  const keep = W[e].q.clone();
  const a = W[up].p, b = W[lo].p;
  const l1 = b.distanceTo(a), l2 = W[e].p.distanceTo(b);
  const d = target.clone().sub(a);
  const dist = THREE.MathUtils.clamp(d.length(), Math.abs(l1 - l2) + 1e-4, (l1 + l2) * 0.999);
  d.normalize();
  const pole = b.clone().sub(a);
  pole.addScaledVector(d, -pole.dot(d));
  if (pole.length() < 1e-4) pole.set(...POLE[end]).addScaledVector(d, -new THREE.Vector3(...POLE[end]).dot(d));
  pole.normalize();
  const x = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist);
  const h = Math.sqrt(Math.max(l1 * l1 - x * x, 0));
  aim(pose, up, lo, a.clone().addScaledVector(d, x).addScaledVector(pole, h));
  aim(pose, lo, e, a.clone().addScaledVector(d, dist));
  setWorldRot(pose, e, keep);
}

const inside = (j, top) => { for (; j >= 0; j = parents[j]) if (j === top) return true; return false; };
const movable = (j) => j === 0 || names[j] in CHAIN;

// one edit from the gizmo, always from the pose as it was when the drag began; a foot stays where it was unless its
// own leg is the one being moved
function apply(pose) {
  if (mode === "translate") {
    if (ctl === 0) pose[0].p.copy(proxy.position);
    else twoBone(pose, names[ctl], proxy.position);
  } else setWorldRot(pose, ctl, proxy.quaternion);
  for (const f of ["LeftFoot", "RightFoot"]) if (!inside(ctl, J[f.replace("Foot", "Leg")])) twoBone(pose, f, drag.pins[f]);
}

// ---- the scene ----
const view = $("#view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
view.append(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15151a);
scene.add(new THREE.HemisphereLight(0xffffff, 0x303040, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-3, 6, 2);
scene.add(sun);
const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 200);
const orbit = new OrbitControls(camera, renderer.domElement);
const tc = new TransformControls(camera, renderer.domElement);
tc.setSize(1.1);
scene.add(tc.getHelper());
const proxy = new THREE.Object3D();
scene.add(proxy);
tc.addEventListener("dragging-changed", (e) => (orbit.enabled = !e.value));

const ball = new THREE.SphereGeometry(1, 18, 12);
const figs = [];
let handles = null;

// the SOMA body (Kimodo's skin, from /api/skin) as one geometry every key's figure shares
function body(S) {
  const arr = (b64, T) => new T(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(arr(S.vertices, Float32Array), 3));
  g.setIndex(new THREE.BufferAttribute(arr(S.faces, Uint32Array), 1));
  g.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(arr(S.skin_index, Uint16Array), 4));
  g.setAttribute("skinWeight", new THREE.BufferAttribute(arr(S.skin_weight, Float32Array), 4));
  g.computeVertexNormals();
  const bind = arr(S.bind, Float32Array), q = arr(S.ref_q, Float32Array);
  return { geo: g,
    inverses: names.map((_, j) => new THREE.Matrix4().fromArray(bind, j * 16).transpose().invert()),
    refInv: names.map((_, j) => new THREE.Quaternion(q[j * 4], q[j * 4 + 1], q[j * 4 + 2], q[j * 4 + 3]).invert()) };
}

// one key's figure: the body on its own flat set of bones, each placed at the joint in world space with Kimodo's turn
function figure(i) {
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.05 });
  const bones = names.map(() => new THREE.Bone());
  const mesh = new THREE.SkinnedMesh(skin.geo, mat);
  mesh.add(...bones);
  mesh.bind(new THREE.Skeleton(bones, skin.inverses), new THREE.Matrix4());
  mesh.frustumCulled = false;
  mesh.userData = { key: i, mat, bones };
  scene.add(mesh);
  return mesh;
}

function draw(i) {
  const W = fk(poses[i]);
  figs[i].userData.bones.forEach((b, j) => {
    b.position.copy(W[j].p);
    b.quaternion.copy(W[j].q).multiply(skin.refInv[j]);
  });
}

function placeHandles() {
  if (!handles) return;
  const W = fk(poses[sel]);
  for (const m of handles.children) if (!(drag && m.userData.joint === ctl)) m.position.copy(W[m.userData.joint].p);
}

function buildHandles() {
  handles?.removeFromParent();
  handles = new THREE.Group();
  const add = (n, color, r) => {
    const m = new THREE.Mesh(ball, new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }));
    m.scale.setScalar(r);
    m.renderOrder = 10;
    m.userData.joint = J[n];
    handles.add(m);
  };
  for (const n of TURN) add(n, 0xffffff, 0.028);
  for (const n of Object.keys(CHAIN)) add(n, 0xff8a3d, 0.05);
  add(names[0], 0xffd23d, 0.065);
  scene.add(handles);
  placeHandles();
}

// a handle clicked: the hands, feet and hips start in move, the other joints in turn
function pick(j) {
  mode = movable(j) ? "translate" : "rotate";
  grab(j);
}

function grab(j) {
  ctl = j;
  const W = fk(poses[sel]);
  proxy.position.copy(W[j].p);
  proxy.quaternion.copy(W[j].q);
  tc.setMode(mode);
  tc.setSpace("world");
  tc.attach(proxy);
  $("#sel").textContent = `K${sel + 1}: ${names[j]}`;
  modeButtons();
}

function setMode(m) {
  if (m === "translate" && ctl != null && !movable(ctl)) return;
  mode = m;
  if (ctl != null) grab(ctl);
  modeButtons();
}

function modeButtons() {
  $("#move").setAttribute("aria-pressed", String(mode === "translate"));
  $("#turn").setAttribute("aria-pressed", String(mode === "rotate"));
}

// pick a key to edit; with `add` (ctrl / shift) the others picked stay picked, for "Only these"
function select(i, add) {
  if (!add) picked.clear();
  picked.add(i);
  figs[i].visible = true;
  sel = i;
  tc.detach();
  ctl = null;
  buildHandles();
  $("#sel").textContent = `K${i + 1}: click a handle`;
  refresh();
}

function refresh() {
  figs.forEach((g, k) => g.userData.mat.color.setHex(k === sel ? COLOR.picked : picked.has(k) ? COLOR.also : k === 0 ? COLOR.start : COLOR.key));
  if (handles) handles.visible = figs[sel].visible;
  if (!figs[sel].visible) tc.detach();
  keyChips();
}

function keyChips() {
  $("#keys").replaceChildren(...poses.map((_, i) => {
    const eye = document.createElement("button");
    eye.className = "eye";
    eye.textContent = figs[i].visible ? "●" : "○";
    eye.title = figs[i].visible ? "Hide" : "Show";
    eye.onclick = () => ((figs[i].visible = !figs[i].visible), refresh());
    const b = document.createElement("button");
    b.className = "key" + (JSON.stringify(toRows(poses[i])) !== origRows[i] ? " edited" : "") + (figs[i].visible ? "" : " hidden");
    b.textContent = `K${i + 1} · ${keyFrames[i] + 1}`;
    b.setAttribute("aria-pressed", String(picked.has(i)));
    b.onclick = (e) => select(i, e.ctrlKey || e.metaKey || e.shiftKey);
    const g = document.createElement("span");
    g.className = "kc";
    g.append(eye, b);
    return g;
  }));
}

// only the picked keys shown, or all of them; then the camera on what is shown
function only(keep) {
  figs.forEach((g, k) => (g.visible = keep(k)));
  refresh();
  frame();
}

// the camera on the shown keys, from where it looks now
function frame() {
  const vis = figs.filter((g) => g.visible);
  const P = (vis.length ? vis : figs).map((g) => fk(poses[g.userData.key])[0].p);
  const bb = new THREE.Box3().setFromPoints(P);
  const c = bb.getCenter(new THREE.Vector3()).setY(0.85);
  const span = Math.max(bb.max.z - bb.min.z, bb.max.x - bb.min.x);
  const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const dist = Math.max((span / 2 + 0.8) / (tan * camera.aspect), 1.2 / tan);
  const dir = camera.position.clone().sub(orbit.target).normalize();
  orbit.target.copy(c);
  camera.position.copy(c).addScaledVector(dir, dist);
}

function changed(i, before) {
  undo.push([i, before]);
  draw(i);
  if (i === sel) placeHandles();
  keyChips();
}

tc.addEventListener("mouseDown", () => {
  const W = fk(poses[sel]);
  drag = { base: clone(poses[sel]), pins: { LeftFoot: W[J.LeftFoot].p.clone(), RightFoot: W[J.RightFoot].p.clone() } };
});
tc.addEventListener("objectChange", () => {
  if (!drag) return;
  const pose = clone(drag.base);
  apply(pose);
  poses[sel] = pose;
  draw(sel);
  placeHandles();
});
tc.addEventListener("mouseUp", () => {
  if (!drag) return;
  const before = drag.base;
  drag = null;
  changed(sel, before);
  grab(ctl); // the gizmo back on the joint (a hand out of reach stops short of where it was dragged)
});

// a click (not an orbit drag, not on the gizmo) picks a handle of the picked key, else a key's figure
const ray = new THREE.Raycaster();
let down = null;
renderer.domElement.addEventListener("pointerdown", (e) => (down = tc.axis ? null : [e.clientX, e.clientY]));
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = handles && ray.intersectObjects(handles.children)[0];
  if (hit) return pick(hit.object.userData.joint);
  const fig = ray.intersectObjects(figs.filter((f) => f.visible))[0];
  if (fig) select(fig.object.userData.key, e.ctrlKey || e.metaKey || e.shiftKey);
});

addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if ((e.ctrlKey || e.metaKey) && e.key === "z") return doUndo(), e.preventDefault();
  if (e.key === "g" || e.key === "G") setMode("translate");
  if (e.key === "r" || e.key === "R") setMode("rotate");
  if (e.key === "o" || e.key === "O") only((k) => picked.has(k));
  if (e.key === "a" || e.key === "A") only(() => true);
  if ((e.key === "h" || e.key === "H") && sel >= 0) ((figs[sel].visible = false), refresh());
  if (e.key === "f" || e.key === "F") frame();
});

function doUndo() {
  const u = undo.pop();
  if (!u) return;
  poses[u[0]] = u[1];
  draw(u[0]);
  select(u[0]);
}

// ---- runs ----
function status(text, bad) {
  $("#status").textContent = text;
  $("#status").className = bad ? "err" : "muted";
}

async function run(engine) {
  for (const b of [$("#block"), $("#kimodo")]) b.disabled = true;
  status(engine === "blocking" ? "Blocking: the keys interpolated and put on the freak in Blender (~15 s)…" : "Kimodo through the keys, then the retarget (~30 s)…");
  try {
    const body = { engine, gpu_ok: $("#gpuok").checked, keys: poses.map((p, i) => ({ frame: keyFrames[i], pose: toRows(p) })) };
    const res = await (await fetch("/api/edit/" + id, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
    if (res.isError) throw new Error(res.structuredContent?.error || res.content?.[0]?.text || "failed");
    const cid = res.structuredContent.clip_id;
    status(`Done: ${cid}${res.structuredContent.warnings?.length ? " (the card says what looks wrong)" : ""}`);
    const links = document.createElement("div");
    links.className = "links";
    links.innerHTML = `<a href="/clip/${cid}" target="_blank">${cid}</a> · <a href="/edit/${cid}">edit this one</a>`;
    const frame = document.createElement("iframe");
    frame.src = "/clip/" + cid;
    $("#result").replaceChildren(links, frame);
    frame.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    status(String(err.message || err), true);
  } finally {
    for (const b of [$("#block"), $("#kimodo")]) b.disabled = false;
  }
}

// ---- start ----
let keyFrames = [];

function resize() {
  const { clientWidth: w, clientHeight: h } = view;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

async function main() {
  $("#clip").textContent = id;
  const res = await fetch("/api/keys/" + id);
  if (!res.ok) throw new Error((await res.text()).trim() || "no key poses for " + id);
  const D = await res.json();
  const sk = await fetch("/api/skin");
  if (!sk.ok) throw new Error((await sk.text()).trim() || "no SOMA body");
  names = D.names;
  parents = D.parents;
  J = Object.fromEntries(names.map((n, i) => [n, i]));
  keyFrames = D.keys.map((k) => k.frame);
  poses = D.keys.map((k) => toPose(k.pose));
  origRows = D.keys.map((k) => JSON.stringify(toRows(toPose(k.pose))));
  skin = body(await sk.json());
  const orig = poses.map(clone);
  poses.forEach((_, i) => figs.push(figure(i)) && draw(i));

  // the hips' track and a ring under each key, the obstacle, a floor under it all
  const hips = poses.map((p) => new THREE.Vector3(p[0].p.x, 0.01, p[0].p.z));
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(hips), new THREE.LineBasicMaterial({ color: 0xf5c542 })));
  const ringGeo = new THREE.RingGeometry(0.17, 0.21, 40).rotateX(-Math.PI / 2);
  hips.forEach((p, i) => {
    const m = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: i ? COLOR.key : COLOR.start }));
    m.position.copy(p);
    scene.add(m);
  });
  const ob = D.meta?.obstacle;
  if (ob) {
    const box = new THREE.Mesh(new THREE.BoxGeometry(1.8, ob.height, ob.depth), new THREE.MeshStandardMaterial({ color: 0x6b5a4e, roughness: 0.85 }));
    box.position.set(0, ob.height / 2, ob.near_z + ob.depth / 2);
    scene.add(box);
  }
  const bb = new THREE.Box3().setFromPoints(hips);
  const c = bb.getCenter(new THREE.Vector3());
  const span = Math.max(bb.max.z - bb.min.z, bb.max.x - bb.min.x, 2);
  const grid = new THREE.GridHelper(Math.ceil(span + 6), Math.ceil(span + 6) * 2, 0x4a4040, 0x26262c);
  grid.position.set(c.x, 0, c.z);
  scene.add(grid);

  resize();
  new ResizeObserver(resize).observe(view);
  orbit.target.set(c.x, 0.85, c.z);
  const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  camera.position.set(c.x - Math.max((span / 2 + 1) / (tan * camera.aspect), 1.4 / tan), 1.3, c.z); // from the body's right
  orbit.update();

  $("#move").onclick = () => setMode("translate");
  $("#turn").onclick = () => setMode("rotate");
  $("#undo").onclick = doUndo;
  $("#only").onclick = () => only((k) => picked.has(k));
  $("#all").onclick = () => only(() => true);
  $("#frame").onclick = frame;
  $("#reset").onclick = () => {
    if (sel < 0) return;
    const before = poses[sel];
    poses[sel] = clone(orig[sel]);
    changed(sel, before);
    if (ctl != null) grab(ctl);
  };
  $("#block").onclick = () => run("blocking");
  $("#kimodo").onclick = () => run("kimodo");
  modeButtons();
  // /edit/<id>#keys=8,9 opens on those keys alone (1-based, as the K buttons count)
  const want = (location.hash.match(/keys=([\d,]+)/)?.[1] || "").split(",").map((k) => k - 1).filter((k) => figs[k]);
  if (want.length) {
    want.forEach((k, i) => select(k, i > 0));
    only((k) => picked.has(k));
  } else select(0);
  renderer.setAnimationLoop(() => {
    orbit.update();
    renderer.render(scene, camera);
  });
}

main().catch((err) => {
  view.textContent = "";
  status(String(err.message || err), true);
});
