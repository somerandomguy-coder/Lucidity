// 3D reconstruction viewer: subject frame W (Y up, X = subject's left, Z = facing).
// Modes: fly (game-style WASD + mouse look), orbit, and the solved photo camera.
import { M4, Lines, Points, TexQuad, Grid } from './gl.js';

const $ = (s) => document.querySelector(s);
const V3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
};
const DEG = 180 / Math.PI;
const DIAG35 = Math.hypot(36, 24);
const LENSES = [13, 26, 52, 77, 120];
const BONES = [[11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [27, 29], [29, 31], [27, 31], [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10]];

let state = null; // one viewer at a time

export function openViewer(photo, assets, onClose) {
  closeViewer();
  const root = $('#viewer');
  root.hidden = false;
  const canvas = $('#glc');
  const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, powerPreference: 'low-power' });
  if (!gl) { root.hidden = true; throw new Error('WebGL2 is not available in this browser'); }
  const r = photo.result, sc = r.scene;
  const det = photo.det;
  const W = det.w, H = det.h, f = r.focal?.f_px ?? det.input[2]; // solved focal (may be estimated)
  const Rwc = sc.R_wc; // rows: X_W, Y_W, Z_W written in C
  const oC = sc.o_c;
  const toW = (pc) => { const d = V3.sub(pc, oC); return [Rwc[0] * d[0] + Rwc[1] * d[1] + Rwc[2] * d[2], Rwc[3] * d[0] + Rwc[4] * d[1] + Rwc[5] * d[2], Rwc[6] * d[0] + Rwc[7] * d[1] + Rwc[8] * d[2]]; };
  const toC = (pw) => [Rwc[0] * pw[0] + Rwc[3] * pw[1] + Rwc[6] * pw[2] + oC[0], Rwc[1] * pw[0] + Rwc[4] * pw[1] + Rwc[7] * pw[2] + oC[1], Rwc[2] * pw[0] + Rwc[5] * pw[1] + Rwc[8] * pw[2] + oC[2]];
  const ray = (u, v) => [(u - W / 2) / f, (v - H / 2) / f, 1];
  // camera axes in W: columns of R_wc. GL camera: right = x_C, up = -y_C, back = -z_C
  const xC = [Rwc[0], Rwc[3], Rwc[6]], yC = [Rwc[1], Rwc[4], Rwc[7]], zC = [Rwc[2], Rwc[5], Rwc[8]];
  const photoCam = { pos: sc.cam_pos_w, right: xC, up: V3.scale(yC, -1), back: V3.scale(zC, -1) };
  const eye = sc.eye_w;
  const f35Photo = photo.intr.f35_mm;

  // ---------- scene content ----------
  const lines = new Lines(gl);
  const helperLines = new Lines(gl);
  const pts = new Points(gl);
  const facePts = new Points(gl);
  const grid = new Grid(gl);
  const joints = sc.joints_w;
  const w8 = sc.joint_w8 || [];
  const segs = [];
  if (joints) {
    const J = (i) => [joints[i * 3], joints[i * 3 + 1], joints[i * 3 + 2]];
    const col = (i, j) => {
      const L = (k) => k >= 11 && k % 2 === 1, R = (k) => k >= 12 && k % 2 === 0;
      const a = Math.min(w8[i] ?? 1, w8[j] ?? 1) > 0.3 ? 1 : 0.35;
      if (L(i) && L(j)) return [1, 0.7, 0.36, a];
      if (R(i) && R(j)) return [0.31, 0.64, 1, a];
      return [0.9, 0.92, 0.95, a];
    };
    for (const [i, j] of BONES) segs.push({ a: J(i), b: J(j), c: col(i, j) });
    const pc = [];
    for (let i = 0; i < 33; i++) pc.push(...((w8[i] ?? 1) > 0.3 ? [1, 1, 1, 1] : [0.6, 0.6, 0.6, 0.5]));
    pts.set(new Float32Array(joints), pc);
  }
  lines.set(segs);
  // face point cloud, coloured from the photo
  if (sc.face_w) {
    const ctx = assets.photo.getContext('2d', { willReadFrequently: true });
    const img = ctx.getImageData(0, 0, assets.w, assets.h).data;
    const cols = [];
    const kx = assets.w / W, ky = assets.h / H;
    for (let i = 0; i < 478; i++) {
      const u = Math.min(assets.w - 1, Math.max(0, Math.round(det.input[214 + i * 3] * kx)));
      const v = Math.min(assets.h - 1, Math.max(0, Math.round(det.input[214 + i * 3 + 1] * ky)));
      const o = (v * assets.w + u) * 4;
      cols.push(img[o] / 255, img[o + 1] / 255, img[o + 2] / 255, 1);
    }
    facePts.set(new Float32Array(sc.face_w), cols);
  }
  // photo cut-out billboard: parallel to the image plane at the subject's depth,
  // so it lines up exactly from the photo camera and looks like a stand-up cut-out elsewhere.
  let subjDepth;
  if (joints) {
    const hip = toC([(joints[23 * 3] + joints[24 * 3]) / 2, (joints[23 * 3 + 1] + joints[24 * 3 + 1]) / 2, (joints[23 * 3 + 2] + joints[24 * 3 + 2]) / 2]);
    subjDepth = hip[2];
  } else {
    subjDepth = toC(eye)[2];
  }
  const cornersAt = (z) => [[0, 0], [W, 0], [0, H], [W, H]].map(([u, v]) => toW(V3.scale(ray(u, v), z)));
  const cutout = assets.cutout ? new TexQuad(gl, assets.cutout, cornersAt(subjDepth)) : null;
  // frustum: photo on an image plane near the camera
  const camDist = V3.len(V3.sub(photoCam.pos, eye));
  const planeZ = Math.min(0.6, Math.max(0.15, camDist * 0.14));
  const photoPlane = new TexQuad(gl, assets.photo, cornersAt(planeZ));
  const fc = cornersAt(planeZ);
  const frustumSegs = [];
  const camCol = [1, 0.7, 0.36, 1];
  for (const c of fc) frustumSegs.push({ a: photoCam.pos, b: c, c: camCol });
  [[0, 1], [1, 3], [3, 2], [2, 0]].forEach(([a, b]) => frustumSegs.push({ a: fc[a], b: fc[b], c: camCol }));
  const camFoot = [photoCam.pos[0], 0, photoCam.pos[2]];
  const eyeFoot = [eye[0], 0, eye[2]];
  frustumSegs.push({ a: photoCam.pos, b: eye, c: [1, 0.85, 0.5, 0.95], dash: 0.08 });
  frustumSegs.push({ a: photoCam.pos, b: camFoot, c: [0.31, 0.64, 1, 0.8], dash: 0.05 });
  frustumSegs.push({ a: camFoot, b: eyeFoot, c: [0.31, 0.64, 1, 0.5], dash: 0.05 });
  frustumSegs.push({ a: [eye[0], eye[1], eye[2]], b: [photoCam.pos[0], eye[1], photoCam.pos[2]], c: [0.7, 0.75, 0.85, 0.45], dash: 0.04 });
  helperLines.set(frustumSegs);

  // ---------- camera state ----------
  const target0 = [eye[0], eye[1] * 0.62, eye[2]];
  const cam = {
    mode: 'fly',
    pos: photoCam.pos.slice(), yaw: 0, pitch: 0, roll: 0,
    f35: f35Photo,
    orbit: { target: target0.slice(), dist: 4, yaw: 0, pitch: 0.25 },
    exact: true, // using the photo camera's exact rotation (incl. roll)
  };
  const setYawPitchFromForward = (fwd) => {
    cam.yaw = Math.atan2(-fwd[0], -fwd[2]);
    cam.pitch = Math.asin(Math.max(-1, Math.min(1, fwd[1])));
  };
  const basis = () => {
    if (cam.exact) return { right: photoCam.right, up: photoCam.up, back: photoCam.back };
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch), cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const fwd = [-sy * cp, sp, -cy * cp];
    const right = V3.norm(V3.cross(fwd, [0, 1, 0]));
    const up = V3.cross(right, fwd);
    return { right, up, back: V3.scale(fwd, -1) };
  };
  const goPhoto = () => { cam.pos = photoCam.pos.slice(); cam.exact = true; cam.f35 = f35Photo; setYawPitchFromForward(V3.scale(photoCam.back, -1)); };
  const startOrbitFromCurrent = () => {
    const o = cam.orbit;
    const d = V3.sub(cam.pos, o.target);
    o.dist = Math.max(0.3, V3.len(d));
    o.yaw = Math.atan2(d[0], d[2]);
    o.pitch = Math.asin(Math.max(-0.99, Math.min(0.99, d[1] / o.dist)));
  };
  const applyOrbit = () => {
    const o = cam.orbit;
    const cp = Math.cos(o.pitch);
    cam.pos = V3.add(o.target, V3.scale([Math.sin(o.yaw) * cp, Math.sin(o.pitch), Math.cos(o.yaw) * cp], o.dist));
    setYawPitchFromForward(V3.norm(V3.sub(o.target, cam.pos)));
    cam.exact = false;
  };
  goPhoto();
  // start in fly mode a bit behind and above the photo camera so both camera and subject are visible
  {
    const back = V3.norm([photoCam.back[0], 0, photoCam.back[2]]);
    const side = V3.cross([0, 1, 0], back);
    cam.pos = V3.add(V3.add(photoCam.pos, V3.scale(back, Math.max(1.2, camDist * 0.6))), V3.add(V3.scale(side, camDist * 0.45), [0, 0.6, 0]));
    cam.pos[1] = Math.max(cam.pos[1], 0.4);
    setYawPitchFromForward(V3.norm(V3.sub(target0, cam.pos)));
    cam.exact = false;
    cam.f35 = Math.min(f35Photo, 26); // wide overview to start; lens buttons change it
  }

  // ---------- UI ----------
  const layers = { cutout: true, skeleton: true, frustum: true };
  const ui = {
    hud: $('#vHud'), help: $('#vHelp'), labels: $('#labels'), opacity: $('#vOpacity'), opacityWrap: $('#vOpacityWrap'),
    dolly: $('#vDolly'), lens: $('#vLens'), pipWrap: $('#vPipWrap'), pip: $('#vPip'),
  };
  // overlay canvas for photo mode (DOM, cheap)
  let overlay = document.getElementById('vPhotoOverlay');
  if (!overlay) { overlay = document.createElement('canvas'); overlay.id = 'vPhotoOverlay'; overlay.style.cssText = 'position:absolute;pointer-events:none;'; root.insertBefore(overlay, ui.labels); }
  overlay.width = assets.w; overlay.height = assets.h;
  overlay.getContext('2d').drawImage(assets.photo, 0, 0);
  // pip
  {
    const s = Math.min(1, 320 / Math.max(assets.w, assets.h));
    ui.pip.width = Math.round(assets.w * s); ui.pip.height = Math.round(assets.h * s);
    ui.pip.getContext('2d').drawImage(assets.photo, 0, 0, ui.pip.width, ui.pip.height);
  }
  ui.lens.innerHTML = [...new Set([Math.round(f35Photo), ...LENSES])].sort((a, b) => a - b)
    .map((l) => `<button type="button" data-lens="${l}" title="${l} mm equivalent">${l === Math.round(f35Photo) ? `${l} (photo)` : l}</button>`).join('');

  const labelEls = {};
  const mkLabel = (key, text, click) => {
    let e = labelEls[key];
    if (!e) {
      e = document.createElement('div'); e.className = 'lbl'; ui.labels.appendChild(e); labelEls[key] = e;
      if (click) { e.style.pointerEvents = 'auto'; e.style.cursor = 'pointer'; e.addEventListener('click', click); }
    }
    e.textContent = text;
    return e;
  };
  const c = r.camera;
  const labelDefs = [
    { key: 'cam', at: photoCam.pos, text: () => `📷 Photo camera · ${c.distance_to_eyes_m.v.toFixed(2)} m`, click: () => setMode('photo') },
    { key: 'eye', at: eye, text: () => `Eyes · elev ${c.elevation_deg.v.toFixed(1)}°` },
    { key: 'h', at: [photoCam.pos[0], photoCam.pos[1] / 2, photoCam.pos[2]], text: () => sc.ground_seen ? `${c.height_above_ground_m.v.toFixed(2)} m high` : `≈${photoCam.pos[1].toFixed(2)} m (assumed floor)` },
    { key: 'd', at: V3.lerp(camFoot, eyeFoot, 0.5), text: () => `${c.horizontal_distance_m.v.toFixed(2)} m` },
  ];

  // ---------- render ----------
  let dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  let vw = 0, vh = 0;
  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    vw = Math.max(1, Math.round(rect.width * dpr)); vh = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
    requestRender();
  };
  /** Projection: photo mode letterboxes the photo exactly; other modes use the lens (35mm eq. on the diagonal). */
  const projection = () => {
    if (cam.mode === 'photo') {
      const s = Math.min(vw / W, vh / H);
      return { P: M4.fromIntrinsics(f * s, f * s, vw / 2, vh / 2, vw, vh), rect: { s, w: W * s, h: H * s } };
    }
    const fpx = cam.f35 * Math.hypot(vw, vh) / DIAG35;
    return { P: M4.fromIntrinsics(fpx, fpx, vw / 2, vh / 2, vw, vh), rect: null };
  };
  let dirty = true, raf = 0;
  function requestRender() { dirty = true; if (!raf) raf = requestAnimationFrame(frame); }

  const keys = new Set();
  let last = performance.now();
  let anim = null; // { t0, dur, from:{pos,f35}, to:{pos,f35} }

  function frame(now) {
    raf = 0;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    let moving = false;
    if (keys.size && cam.mode !== 'photo') moving = stepKeys(dt);
    if (anim) {
      const t = Math.min(1, (now - anim.t0) / anim.dur);
      const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      cam.pos = V3.lerp(anim.from.pos, anim.to.pos, e);
      cam.f35 = anim.from.f35 + (anim.to.f35 - anim.from.f35) * e;
      if (anim.aim) setYawPitchFromForward(V3.norm(V3.lerp(anim.from.fwd, V3.norm(V3.sub(eye, cam.pos)), Math.min(1, e * 1.6))));
      if (t >= 1) { anim = null; syncLensButtons(); }
      moving = true;
    }
    if (dirty || moving) { draw(); dirty = false; }
    if (moving || keys.size || anim) raf = requestAnimationFrame(frame);
  }

  function draw() {
    gl.viewport(0, 0, vw, vh);
    gl.clearColor(0.043, 0.051, 0.07, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const b = basis();
    const V = M4.view(cam.pos, b.right, b.up, b.back);
    const { P, rect } = projection();
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    // opaque-ish first
    if (layers.cutout && cutout) cutout.draw(V, P, 0.5, 1);
    if (layers.skeleton) { lines.draw(V, P, vw, vh, 3 * dpr); pts.draw(V, P, 3 * dpr); }
    facePts.draw(V, P, 2.2 * dpr);
    if (layers.frustum && cam.mode !== 'photo') { photoPlane.draw(V, P, 0, 0.92); helperLines.draw(V, P, vw, vh, 1.8 * dpr); }
    // translucent ground last (no depth write)
    gl.depthMask(false);
    grid.draw(V, P, cam.pos, sc.ground_seen ? 1 : 0.55);
    gl.depthMask(true);
    // photo overlay (photo mode)
    if (rect) {
      const cssW = rect.w / dpr, cssH = rect.h / dpr;
      Object.assign(overlay.style, { display: 'block', width: `${cssW}px`, height: `${cssH}px`, left: `calc(50% - ${cssW / 2}px)`, top: `calc(50% - ${cssH / 2}px)`, opacity: ui.opacity.value });
    } else overlay.style.display = 'none';
    updateLabels(V, P);
    updateHud();
  }

  function updateLabels(V, P) {
    const show = layers.frustum && cam.mode !== 'photo';
    const VP = M4.mul(P, V);
    for (const d of labelDefs) {
      const e = mkLabel(d.key, d.text(), d.click);
      const p = d.at;
      const x = VP[0] * p[0] + VP[4] * p[1] + VP[8] * p[2] + VP[12];
      const y = VP[1] * p[0] + VP[5] * p[1] + VP[9] * p[2] + VP[13];
      const w = VP[3] * p[0] + VP[7] * p[1] + VP[11] * p[2] + VP[15];
      if (!show || w <= 0.05 || (d.key === 'h' && !show)) { e.style.display = 'none'; continue; }
      const sx = (x / w * 0.5 + 0.5) * (vw / dpr), sy = (1 - (y / w * 0.5 + 0.5)) * (vh / dpr);
      if (sx < -50 || sy < -20 || sx > vw / dpr + 50 || sy > vh / dpr + 40) { e.style.display = 'none'; continue; }
      e.style.display = 'block';
      e.style.left = `${sx}px`; e.style.top = `${sy - 8}px`;
    }
  }

  function updateHud() {
    const b = basis();
    const fwd = V3.scale(b.back, -1);
    const hEye = cam.pos[1] - eye[1];
    const dvec = V3.sub(cam.pos, eye);
    const dh = Math.hypot(dvec[0], dvec[2]);
    const dist = V3.len(dvec);
    const pitch = -Math.asin(Math.max(-1, Math.min(1, fwd[1]))) * DEG;
    const elev = Math.atan2(hEye, dh) * DEG;
    let az = Math.atan2(dvec[0], dvec[2]) * DEG;
    if (photo.mirrored) az = -az;
    // roll: angle of the camera's right vector out of the horizontal plane
    const roll = cam.exact ? c.roll_deg.v : 0;
    const row = (k, v, ref, unit, d = 2) => {
      const delta = v - ref;
      const ds = Math.abs(delta) < (unit === '°' ? 0.5 : 0.01) ? '' : ` <span class="delta">${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(unit === '°' ? 0 : d)}</span>`;
      return `<tr><td>${k}</td><td>${v.toFixed(unit === '°' ? 1 : d)}${unit}${ds}</td></tr>`;
    };
    const title = cam.mode === 'photo' ? 'Photo camera (solved)' : 'Virtual camera';
    ui.hud.innerHTML = `<h4>${title}</h4><table>
      ${row('Height vs. eyes', hEye, c.height_above_eye_m.v, ' m')}
      ${row('Height above floor', cam.pos[1], photoCam.pos[1], ' m')}
      ${row('Distance to eyes', dist, c.distance_to_eyes_m.v, ' m')}
      ${row('Tilt (+ down)', pitch, c.pitch_deg.v, '°')}
      ${row('Elevation', elev, c.elevation_deg.v, '°')}
      ${row('Side angle', az, c.azimuth_deg.v, '°')}
      ${cam.exact ? row('Roll', roll, c.roll_deg.v, '°') : ''}
      <tr><td>Lens</td><td>${cam.f35.toFixed(0)} mm eq.</td></tr></table>`;
  }

  // ---------- controls ----------
  function stepKeys(dt) {
    const fast = keys.has('shift') ? 3 : 1;
    const speed = 1.6 * fast * dt;
    const b = basis();
    const fwd = V3.norm([-b.back[0], 0, -b.back[2]]);
    const right = V3.norm([b.right[0], 0, b.right[2]]);
    let m = [0, 0, 0];
    if (keys.has('w') || keys.has('arrowup')) m = V3.add(m, fwd);
    if (keys.has('s') || keys.has('arrowdown')) m = V3.sub(m, fwd);
    if (keys.has('d') || keys.has('arrowright')) m = V3.add(m, right);
    if (keys.has('a') || keys.has('arrowleft')) m = V3.sub(m, right);
    if (keys.has('e') || keys.has(' ')) m = V3.add(m, [0, 1, 0]);
    if (keys.has('q') || keys.has('c')) m = V3.sub(m, [0, 1, 0]);
    if (V3.len(m) < 1e-6) return false;
    breakExact();
    cam.pos = V3.add(cam.pos, V3.scale(V3.norm(m), speed));
    if (cam.mode === 'orbit') { cam.orbit.target = V3.add(cam.orbit.target, V3.scale(V3.norm(m), speed)); }
    return true;
  }
  function breakExact() {
    if (cam.exact) { setYawPitchFromForward(V3.scale(photoCam.back, -1)); cam.exact = false; }
    if (cam.mode === 'photo') setMode('fly', true);
  }

  function setMode(m, keepPose = false) {
    const prev = cam.mode;
    cam.mode = m;
    root.querySelectorAll('[data-mode]').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
    ui.opacityWrap.hidden = m !== 'photo';
    ui.pipWrap.hidden = m === 'photo';
    if (m === 'photo') { anim = null; goPhoto(); }
    if (m === 'orbit') { if (prev === 'photo' && !keepPose) breakExact(); startOrbitFromCurrent(); }
    if (m === 'fly' && prev === 'photo' && !keepPose) { /* keep the photo pose, just unlock */ }
    syncLensButtons();
    ui.help.innerHTML = m === 'fly'
      ? '<b>Fly</b>: drag to look · W A S D move · Q / E down / up · Shift faster · wheel moves forward'
      : m === 'orbit' ? '<b>Orbit</b>: drag to orbit · right-drag or Shift-drag to pan · wheel to zoom'
        : '<b>Photo camera</b>: the 3D scene from the solved camera over the photo. Use the slider to compare. Drag to fly away from here.';
    requestRender();
  }
  function syncLensButtons() {
    ui.lens.querySelectorAll('button').forEach((b) => b.classList.toggle('on', Math.abs(+b.dataset.lens - cam.f35) < 0.6));
  }
  function setLens(f35) {
    if (cam.mode === 'photo') setMode('fly', true);
    breakExact();
    const from = { pos: cam.pos.slice(), f35: cam.f35, fwd: V3.scale(basis().back, -1) };
    let toPos = cam.pos.slice();
    const dolly = ui.dolly.checked;
    if (dolly) {
      // keep the subject the same size: d' = d · f'/f along the line to the eyes
      toPos = V3.add(eye, V3.scale(V3.sub(cam.pos, eye), f35 / cam.f35));
      if (cam.mode === 'orbit') cam.orbit.dist *= f35 / cam.f35;
    }
    // a dolly zoom only makes sense looking at the subject, so aim at the eyes while moving
    anim = { t0: performance.now(), dur: dolly ? 900 : 250, from, to: { pos: toPos, f35 }, aim: dolly && cam.mode !== 'orbit' };
    ui.lens.querySelectorAll('button').forEach((b) => b.classList.toggle('on', +b.dataset.lens === f35));
    requestRender();
  }

  // pointer
  let drag = null;
  const onDown = (e) => {
    canvas.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, btn: e.button, shift: e.shiftKey, id: e.pointerId };
    canvas.focus();
  };
  const pointers = new Map();
  const onMove = (e) => {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (!drag || drag.id !== e.pointerId) return;
    if (pointers.size >= 2) return; // pinch handled separately
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (cam.mode === 'photo') { breakExact(); }
    if (cam.mode === 'orbit') {
      const o = cam.orbit;
      if (drag.btn === 2 || drag.shift) {
        const b = basis();
        const k = o.dist * 0.0015;
        o.target = V3.add(o.target, V3.add(V3.scale(b.right, -dx * k), V3.scale(b.up, dy * k)));
      } else {
        o.yaw -= dx * 0.006;
        o.pitch = Math.max(-1.45, Math.min(1.45, o.pitch + dy * 0.006));
      }
      applyOrbit();
    } else {
      breakExact();
      cam.yaw -= dx * 0.0035;
      cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch - dy * 0.0035));
    }
    requestRender();
  };
  const onUp = (e) => { pointers.delete(e.pointerId); if (drag && drag.id === e.pointerId) drag = null; pinch = null; };
  let pinch = null;
  const onTouchMove = () => {
    if (pointers.size !== 2) return;
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    if (pinch != null) zoomBy((pinch - d) * 4);
    pinch = d;
  };
  const zoomBy = (delta) => {
    if (cam.mode === 'photo') breakExact();
    if (cam.mode === 'orbit') {
      cam.orbit.dist = Math.max(0.3, Math.min(40, cam.orbit.dist * Math.exp(delta * 0.0012)));
      applyOrbit();
    } else {
      breakExact();
      const b = basis();
      cam.pos = V3.add(cam.pos, V3.scale(b.back, delta * 0.002));
    }
    requestRender();
  };
  const onWheel = (e) => { e.preventDefault(); zoomBy(e.deltaY * (e.deltaMode === 1 ? 30 : 1)); };
  const keyName = (e) => (e.key === ' ' ? ' ' : e.key.toLowerCase());
  const MOVE_KEYS = new Set(['w', 'a', 's', 'd', 'q', 'e', 'c', ' ', 'shift', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']);
  const onKeyDown = (e) => {
    if (e.target.matches?.('input, select, textarea')) return;
    const k = keyName(e);
    if (k === 'escape') { closeViewer(); return; }
    if (k === '1') setMode('fly'); else if (k === '2') setMode('orbit'); else if (k === '3') setMode('photo');
    if (!MOVE_KEYS.has(k)) return;
    e.preventDefault();
    keys.add(k);
    last = performance.now();
    requestRender();
  };
  const onKeyUp = (e) => keys.delete(keyName(e));
  const onBlur = () => keys.clear();

  canvas.tabIndex = 0;
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', (e) => { onMove(e); if (e.pointerType === 'touch') onTouchMove(); });
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('dblclick', () => setMode('photo'));
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', onBlur);
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);

  const onTop = (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.mode) setMode(b.dataset.mode);
    if (b.dataset.lens) setLens(+b.dataset.lens);
    if (b.dataset.layer) { layers[b.dataset.layer] = !layers[b.dataset.layer]; b.classList.toggle('on', layers[b.dataset.layer]); requestRender(); }
  };
  const top = root.querySelector('.v-top');
  top.addEventListener('click', onTop);
  const onOpacity = () => requestRender();
  ui.opacity.addEventListener('input', onOpacity);
  const closeBtn = $('#vClose');
  const onCloseClick = () => closeViewer();
  closeBtn.addEventListener('click', onCloseClick);
  // cut-out button only when we have one
  const cutBtn = root.querySelector('[data-layer="cutout"]');
  cutBtn.hidden = !cutout;
  root.querySelectorAll('[data-layer]').forEach((b) => b.classList.toggle('on', layers[b.dataset.layer]));

  state = {
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
      top.removeEventListener('click', onTop);
      ui.opacity.removeEventListener('input', onOpacity);
      closeBtn.removeEventListener('click', onCloseClick);
      // fresh canvas next time drops all GL resources and listeners
      const fresh = canvas.cloneNode(false);
      canvas.replaceWith(fresh);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      ui.labels.innerHTML = '';
      overlay.style.display = 'none';
      root.hidden = true;
      onClose?.();
    },
    // exposed for tests / debugging
    debug: { cam, photoCam, projection, basis, toW, ray, W, H, f, setMode, get vw() { return vw; }, get vh() { return vh; } },
  };
  window.__camposeViewer = state.debug;
  setMode('fly', true);
  resize();
  canvas.focus();
  return state;
}

export function closeViewer() {
  if (!state) return;
  const s = state;
  state = null;
  s.dispose();
}
