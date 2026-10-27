// The clip player: the freak's preview GLB (Rig_* joint nodes, rest at zero rotation, exported Y up) driven frame by
// frame from the clip's quaternions. Clip data is in Blender axes (Z up, facing +Y, right +X); glTF is Y up with the
// character facing -Z, so (x, y, z) -> (x, z, -y) for vectors and quaternion axes alike.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const VIEWS = { side: [1, 0.18, 0], "3/4": [0.72, 0.22, -0.72], front: [0, 0.16, -1] };

export class Player {
  constructor(stage) {
    const renderer = (this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }));
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NeutralToneMapping;
    const scene = (this.scene = new THREE.Scene());
    scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
    const key = new THREE.DirectionalLight(0xfff4e8, 1.2);
    key.position.set(4, 8, -5);
    const rim = new THREE.DirectionalLight(0xffb48a, 0.8);
    rim.position.set(-5, 4, 6);
    scene.add(key, rim);
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 500);
    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.enableZoom = false; // the chat scrolls; ctrl/cmd + wheel zooms
    renderer.domElement.addEventListener("wheel", (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const off = this.camera.position.clone().sub(this.controls.target);
      this.camera.position.copy(this.controls.target).add(off.multiplyScalar(Math.exp(Math.sign(e.deltaY) * 0.12)));
    }, { passive: false });
    this.speed = 1;
    this.playing = true;
    this.t = 0;
    this.onframe = null;
    this.attach(stage);
    let last = performance.now();
    const tick = (now) => {
      requestAnimationFrame(tick);
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      if (!this.clip || !this.visible) return;
      if (this.playing) {
        const len = this.clip.frames / this.clip.fps;
        this.t += dt * this.speed;
        if (this.t > len + (this.clip.loop ? 0 : 0.6)) this.t = 0; // a one-shot holds its last pose a moment
        this.show(Math.min(Math.floor(this.t * this.clip.fps), this.clip.frames - 1));
      }
      this.controls.update();
      renderer.render(scene, this.camera);
    };
    requestAnimationFrame(tick);
  }

  attach(stage) {
    this.stage = stage;
    stage.prepend(this.renderer.domElement);
    this.ro?.disconnect();
    this.io?.disconnect();
    (this.ro = new ResizeObserver(() => this.resize())).observe(stage);
    (this.io = new IntersectionObserver(([e]) => (this.visible = e.isIntersecting))).observe(stage);
    this.visible = true;
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.stage;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.clip) this.view(this.current);
  }

  // props: boxes such as the vault's obstacle, {forward, depth, height, width} in studs from the body's start spot
  setProps(props) {
    this.props?.removeFromParent();
    this.props = new THREE.Group();
    const start = this.hipsAt(this.clip.off[0]);
    for (const p of props || []) {
      const box = new THREE.Mesh(new THREE.BoxGeometry(p.width, p.height, p.depth),
        new THREE.MeshStandardMaterial({ color: 0x6b5a4e, roughness: 0.85 }));
      box.position.set(start.x, p.height / 2, start.z - p.forward - p.depth / 2);
      this.props.add(box);
    }
    this.scene.add(this.props);
  }

  async load(rig, clip) {
    const bin = atob(rig.data);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const root = (await new GLTFLoader().parseAsync(bytes.buffer, "")).scene;
    if (this.model) this.scene.remove(this.model);
    this.model = root;
    this.scene.add(root);
    this.nodes = clip.joints.map((n) => root.getObjectByName(n));
    const missing = clip.joints.filter((n, i) => !this.nodes[i]);
    if (missing.length) throw new Error("the rig has no " + missing.join(", "));
    this.hips0 = this.nodes[0].position.clone();
    this.height = new THREE.Box3().setFromObject(root).max.y; // at rest, feet on the floor (y 0)
    this.clip = clip;
    // the floor: a 1-stud grid under the whole travel, the start spot marked
    const P = clip.off.map((o) => this.hipsAt(o));
    const box = new THREE.Box3().setFromPoints(P);
    const c = box.getCenter(new THREE.Vector3());
    const span = Math.max(box.max.x - box.min.x, box.max.z - box.min.z);
    const size = Math.ceil(span + 14);
    this.grid?.removeFromParent();
    this.grid = new THREE.GridHelper(size, size, 0x5a4a44, 0x2c2c33);
    this.grid.position.set(Math.round(c.x), 0, Math.round(c.z));
    this.scene.add(this.grid);
    this.focus = new THREE.Vector3(c.x, this.height * 0.5, c.z);
    this.span = span;
    this.view("side");
    this.show(0);
  }

  hipsAt(o) {
    return this.hips0.clone().add(new THREE.Vector3(o[0], o[2], -o[1]));
  }

  // the whole body over the whole travel in frame: 1.3 x its height, and its travel plus a body's width across
  view(name) {
    this.current = name;
    const d = VIEWS[name] || VIEWS.side;
    const tan = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const dist = Math.max((this.height * 0.65) / tan, ((this.span + this.height * 1.4) / 2) / (tan * this.camera.aspect));
    this.camera.position.copy(this.focus).add(new THREE.Vector3(...d).normalize().multiplyScalar(dist));
    this.controls.target.copy(this.focus);
    this.controls.update();
  }

  // the key poses standing along the path, as in MotionBricks' own demo: a flat-coloured copy of the freak at each key
  // (the first green, the rest blue), a ring on the floor under each and the hips' track in yellow
  setKeys(keys) {
    this.keyGroup?.removeFromParent();
    const g = (this.keyGroup = new THREE.Group());
    const c = this.clip;
    const floor = (o) => this.hipsAt(o).setY(0.02);
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(c.off.map(floor)),
      new THREE.LineBasicMaterial({ color: 0xf5c542 })));
    const rings = [0x3ddc84, 0x4aa8ff].map((color) => new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
    this.ghosts = keys.map((f, i) => {
      f = Math.min(f, c.frames - 1);
      const ghost = this.model.clone();
      const mat = new THREE.MeshStandardMaterial({ color: rings[i ? 1 : 0].color, roughness: 0.7, transparent: true });
      ghost.traverse((o) => o.isMesh && (o.material = mat));
      ghost.userData.mat = mat;
      this.pose(c.joints.map((n) => ghost.getObjectByName(n)), f);
      const ring = new THREE.Mesh(new THREE.RingGeometry(1.0, 1.25, 40), rings[i ? 1 : 0]);
      ring.rotation.x = -Math.PI / 2;
      ring.position.copy(floor(c.off[f]));
      g.add(ghost, ring);
      return ghost;
    });
    this.scene.add(g);
    this.focusKey(-1);
  }

  // one key pose solid and the others faint (they crowd where the keys come fast); -1 shows them all alike
  focusKey(k) {
    this.ghosts?.forEach((g, i) => (g.userData.mat.opacity = k < 0 ? 0.75 : i === k ? 0.95 : 0.18));
  }

  pose(nodes, f) {
    const c = this.clip;
    c.q[f].forEach(([w, x, y, z], i) => nodes[i].quaternion.set(x, z, -y, w));
    nodes[0].position.copy(this.hipsAt(c.off[f]));
  }

  show(f) {
    this.frame = f;
    this.pose(this.nodes, f);
    this.onframe?.(f);
  }

  seek(f) {
    this.t = f / this.clip.fps;
    this.show(f);
  }
}
