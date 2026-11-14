// 3D reconstruction viewer, in subject frame W (Y up, X = subject's left, Z = facing).
//
// Three views:
//   Overview      orbit around the scene; the virtual camera is drawn and its picture
//                 is shown live in the corner (picture-in-picture)
//   Camera view   look through the virtual camera (game-style: drag to look, WASD to move)
//   Original shot the solved camera of the photo, with the photo overlaid
// The virtual camera is driven by sliders, keyboard, mouse or touch, and its numbers
// (tilt, height vs eyes, distance, side angle) update live.
import { M4, Lines, Points, TexQuad, Grid, Mesh } from './gl.js';
import { buildMannequin } from './mannequin.js';

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
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

let state = null;

export function openViewer(photo, assets, onClose) {
  closeViewer();
  const root = $('#viewer');
  root.hidden = false;
  const canvas = $('#glc');
  const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, powerPreference: 'low-power' });
  if (!gl) { root.hidden = true; throw new Error('WebGL2 is not available in this browser'); }
  const r = photo.result, sc = r.scene, det = photo.det, c = r.camera;
  const W = det.w, H = det.h, f = r.focal?.f_px ?? det.input[2];
  const Rwc = sc.R_wc, oC = sc.o_c;
  const toW = (pc) => { const d = V3.sub(pc, oC); return [Rwc[0] * d[0] + Rwc[1] * d[1] + Rwc[2] * d[2], Rwc[3] * d[0] + Rwc[4] * d[1] + Rwc[5] * d[2], Rwc[6] * d[0] + Rwc[7] * d[1] + Rwc[8] * d[2]]; };
  const toC = (pw) => [Rwc[0] * pw[0] + Rwc[3] * pw[1] + Rwc[6] * pw[2] + oC[0], Rwc[1] * pw[0] + Rwc[4] * pw[1] + Rwc[7] * pw[2] + oC[1], Rwc[2] * pw[0] + Rwc[5] * pw[1] + Rwc[8] * pw[2] + oC[2]];
  const ray = (u, v) => [(u - W / 2) / f, (v - H / 2) / f, 1];
  const xC = [Rwc[0], Rwc[3], Rwc[6]], yC = [Rwc[1], Rwc[4], Rwc[7]], zC = [Rwc[2], Rwc[5], Rwc[8]];
  const photoCam = { pos: sc.cam_pos_w, right: xC, up: V3.scale(yC, -1), back: V3.scale(zC, -1) };
  const eye = sc.eye_w;
  const f35Photo = photo.intr.f35_mm;
  const aspect = W / H;

  // ---------- static scene ----------
  const skel = new Lines(gl), pts = new Points(gl), facePts = new Points(gl), grid = new Grid(gl);
  const photoLines = new Lines(gl), virtLines = new Lines(gl), body = new Mesh(gl);
  const joints = sc.joints_w, w8 = sc.joint_w8 || [];
  if (joints) {
    const J = (i) => [joints[i * 3], joints[i * 3 + 1], joints[i * 3 + 2]];
    const col = (i, j) => {
      const L = (k) => k >= 11 && k % 2 === 1, R = (k) => k >= 12 && k % 2 === 0;
      const a = Math.min(w8[i] ?? 1, w8[j] ?? 1) > 0.3 ? 1 : 0.35;
      if (L(i) && L(j)) return [1, 0.7, 0.36, a];
      if (R(i) && R(j)) return [0.31, 0.64, 1, a];
      return [0.9, 0.92, 0.95, a];
    };
    skel.set(BONES.map(([i, j]) => ({ a: J(i), b: J(j), c: col(i, j) })));
    const pc = [];
    for (let i = 0; i < 33; i++) pc.push(...((w8[i] ?? 1) > 0.3 ? [1, 1, 1, 1] : [0.6, 0.6, 0.6, 0.5]));
    pts.set(new Float32Array(joints), pc);
  }
  if (sc.face_w) {
    const ctx = assets.photo.getContext('2d', { willReadFrequently: true });
    const img = ctx.getImageData(0, 0, assets.w, assets.h).data;
    const cols = [];
    const kx = assets.w / W, ky = assets.h / H;
    for (let i = 0; i < 478; i++) {
      const u = clamp(Math.round(det.input[214 + i * 3] * kx), 0, assets.w - 1);
      const v = clamp(Math.round(det.input[214 + i * 3 + 1] * ky), 0, assets.h - 1);
      const o = (v * assets.w + u) * 4;
      cols.push(img[o] / 255, img[o + 1] / 255, img[o + 2] / 255, 1);
    }
    facePts.set(new Float32Array(sc.face_w), cols);
  }
  body.set(buildMannequin(joints, w8, sc.face_w));
  // photo cut-out parallel to the image plane at the subject's depth
  const subjDepth = joints
    ? toC([(joints[69] + joints[72]) / 2, (joints[70] + joints[73]) / 2, (joints[71] + joints[74]) / 2])[2]
    : toC(eye)[2];
  const cornersAt = (z) => [[0, 0], [W, 0], [0, H], [W, H]].map(([u, v]) => toW(V3.scale(ray(u, v), z)));
  const cutout = assets.cutout ? new TexQuad(gl, assets.cutout, cornersAt(subjDepth)) : null;
  const camDist = V3.len(V3.sub(photoCam.pos, eye));
  const planeZ = clamp(camDist * 0.14, 0.12, 0.6);
  const photoPlane = new TexQuad(gl, assets.photo, cornersAt(planeZ));
  const camFoot = [photoCam.pos[0], 0, photoCam.pos[2]];
  const eyeFoot = [eye[0], 0, eye[2]];
  {
    const fc = cornersAt(planeZ), col = [1, 0.7, 0.36, 1], segs = [];
    for (const q of fc) segs.push({ a: photoCam.pos, b: q, c: col });
    [[0, 1], [1, 3], [3, 2], [2, 0]].forEach(([a, b]) => segs.push({ a: fc[a], b: fc[b], c: col }));
    segs.push({ a: photoCam.pos, b: eye, c: [1, 0.85, 0.5, 0.9], dash: 0.08 });
    segs.push({ a: photoCam.pos, b: camFoot, c: [1, 0.7, 0.36, 0.6], dash: 0.05 });
    segs.push({ a: camFoot, b: eyeFoot, c: [1, 0.7, 0.36, 0.35], dash: 0.05 });
    photoLines.set(segs);
  }

  // ---------- cameras ----------
  /** Orientation from yaw (around +Y, 0 = looking down -Z), pitch (+ up) and roll. */
  const basisFrom = (yaw, pitch, roll) => {
    const cp = Math.cos(pitch);
    const fwd = [-Math.sin(yaw) * cp, Math.sin(pitch), -Math.cos(yaw) * cp];
    const r0 = V3.norm(V3.cross(fwd, [0, 1, 0]));
    const right = V3.add(V3.scale(r0, Math.cos(roll)), V3.scale(V3.cross(fwd, r0), Math.sin(roll)));
    const up = V3.cross(right, fwd);
    return { right, up, back: V3.scale(fwd, -1), fwd };
  };
  const anglesFrom = (fwd, right) => {
    const yaw = Math.atan2(-fwd[0], -fwd[2]);
    const pitch = Math.asin(clamp(fwd[1], -1, 1));
    const r0 = V3.norm(V3.cross(fwd, [0, 1, 0]));
    const roll = right ? Math.atan2(V3.dot(V3.cross(r0, right), fwd), V3.dot(r0, right)) : 0;
    return { yaw, pitch, roll };
  };
  /** Same metrics as the Rust solver, for any camera pose in W. */
  const metricsOf = (pos, b) => {
    const U = [b.right[1], -b.up[1], b.fwd[1]]; // world up in camera coords (x right, y down, z fwd)
    const pitch = Math.asin(clamp(-U[2], -1, 1)) * DEG;
    const roll = Math.atan2(U[0], -U[1]) * DEG;
    const d = V3.sub(pos, eye);
    const dh = Math.hypot(d[0], d[2]);
    let az = Math.atan2(d[0], d[2]) * DEG;
    if (photo.mirrored) az = -az;
    return { pitch, roll, hEye: d[1], hFloor: pos[1], dist: V3.len(d), dh, elev: Math.atan2(d[1], dh) * DEG, az };
  };

  // virtual camera rig, starts exactly at the solved photo camera
  const pa = anglesFrom(V3.scale(photoCam.back, -1), photoCam.right);
  const vcam = { pos: photoCam.pos.slice(), yaw: pa.yaw, pitch: pa.pitch, roll: pa.roll, f35: f35Photo, aim: false };
  const vBasis = () => basisFrom(vcam.yaw, vcam.pitch, vcam.roll);
  const aimAtEyes = () => {
    const a = anglesFrom(V3.norm(V3.sub(eye, vcam.pos)));
    vcam.yaw = a.yaw; vcam.pitch = a.pitch;
  };
  // overview orbit camera
  const target0 = [eye[0], eye[1] * 0.6, eye[2]];
  const orbit = { target: target0.slice(), dist: Math.max(3, camDist * 1.6), yaw: 0, pitch: 0.35 };
  {
    // start beside the photo camera, rotated a bit, so subject and camera are both in view
    const d = V3.sub(photoCam.pos, target0);
    orbit.yaw = Math.atan2(d[0], d[2]) + 0.9;
  }
  const orbitPos = () => {
    const cp = Math.cos(orbit.pitch);
    return V3.add(orbit.target, V3.scale([Math.sin(orbit.yaw) * cp, Math.sin(orbit.pitch), Math.cos(orbit.yaw) * cp], orbit.dist));
  };
  const orbitBasis = () => {
    const a = anglesFrom(V3.norm(V3.sub(orbit.target, orbitPos())));
    return basisFrom(a.yaw, a.pitch, 0);
  };

  // ---------- UI ----------
  let mode = 'orbit';
  const layers = { mannequin: true, cutout: false, skeleton: false };
  const ui = {
    labels: $('#labels'), opacity: $('#vOpacity'), opacityWrap: $('#vOpacityWrap'), help: $('#vHelp'),
    panel: $('#vPanel'), hud: $('#vHud'), lens: $('#vLens'), dolly: $('#vDolly'), aim: $('#vAim'),
    pip: $('#vPip'), pipLabel: $('#vPipLabel'),
    s: { az: $('#sAz'), dist: $('#sDist'), h: $('#sH'), tilt: $('#sTilt'), roll: $('#sRoll') },
    o: { az: $('#oAz'), dist: $('#oDist'), h: $('#oH'), tilt: $('#oTilt'), roll: $('#oRoll') },
  };
  const resetToPhoto = () => {
    vcam.pos = photoCam.pos.slice();
    Object.assign(vcam, { yaw: pa.yaw, pitch: pa.pitch, roll: pa.roll, f35: f35Photo, aim: false });
    ui.aim.checked = false;
  };
  ui.aim.checked = false;
  ui.s.dist.max = String(Math.max(10, Math.ceil(camDist * 2.5)));
  ui.s.h.max = String(Math.max(4, Math.ceil(photoCam.pos[1] + 1.5)));
  ui.lens.innerHTML = [...new Set([Math.round(f35Photo), ...LENSES])].sort((a, b) => a - b)
    .map((l) => `<button type="button" data-lens="${l}" title="${l} mm equivalent">${l === Math.round(f35Photo) ? `${l}·photo` : l}</button>`).join('');
  // photo overlay (Original shot) and the photo picture-in-picture (Camera view) are plain canvases
  const mkCanvas = (id, parent, css) => {
    let e = document.getElementById(id);
    if (!e) { e = document.createElement('canvas'); e.id = id; e.style.cssText = css; parent.appendChild(e); }
    e.width = assets.w; e.height = assets.h;
    e.getContext('2d').drawImage(assets.photo, 0, 0);
    return e;
  };
  const overlay = mkCanvas('vPhotoOverlay', root, 'position:absolute;pointer-events:none;display:none;');
  root.insertBefore(overlay, ui.labels);
  const pipPhoto = mkCanvas('vPipPhoto', ui.pip, 'display:none;');

  const labelEls = {};
  const mkLabel = (key, text, click) => {
    let e = labelEls[key];
    if (!e) {
      e = document.createElement('div'); e.className = 'lbl'; ui.labels.appendChild(e); labelEls[key] = e;
      if (click) { e.style.pointerEvents = 'auto'; e.style.cursor = 'pointer'; e.addEventListener('click', click); }
    }
    if (e.textContent !== text) e.textContent = text;
    return e;
  };

  // ---------- render ----------
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  let vw = 1, vh = 1;
  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    vw = Math.max(1, Math.round(rect.width * dpr)); vh = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
    requestRender();
  };
  const lensProj = (f35, w, h) => { const fpx = f35 * Math.hypot(w, h) / DIAG35; return M4.fromIntrinsics(fpx, fpx, w / 2, h / 2, w, h); };
  /** Virtual-camera picture: lens defined on a W:H frame fitted in the viewport, like the photo. */
  const virtProj = (w, h) => {
    const s = Math.min(w / W, h / H);
    const fpx = vcam.f35 * Math.hypot(W * s, H * s) / DIAG35;
    return M4.fromIntrinsics(fpx, fpx, w / 2, h / 2, w, h);
  };

  let dirty = true, raf = 0;
  function requestRender() { dirty = true; if (!raf) raf = requestAnimationFrame(frame); }
  const keys = new Set();
  let last = performance.now();
  let anim = null;

  function frame(now) {
    raf = 0;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    let moving = false;
    if (keys.size) moving = stepKeys(dt);
    if (anim) {
      const t = Math.min(1, (now - anim.t0) / anim.dur);
      const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      vcam.pos = V3.lerp(anim.from.pos, anim.to.pos, e);
      vcam.f35 = anim.from.f35 + (anim.to.f35 - anim.from.f35) * e;
      if (anim.aim || vcam.aim) aimAtEyes();
      if (t >= 1) { anim = null; syncLens(); }
      moving = true;
    }
    if (dirty || moving) { draw(); dirty = false; }
    if (moving || keys.size || anim) raf = requestAnimationFrame(frame);
  }

  function drawScene(V, P, camPos, opts) {
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    if (layers.mannequin) body.draw(V, P);
    if (layers.cutout && cutout) cutout.draw(V, P, 0.5, 1);
    if (layers.skeleton) { skel.draw(V, P, opts.w, opts.h, 3 * dpr); pts.draw(V, P, 3 * dpr); facePts.draw(V, P, 2 * dpr); }
    if (opts.photoCam) { photoPlane.draw(V, P, 0, 0.92); photoLines.draw(V, P, opts.w, opts.h, 1.8 * dpr); }
    if (opts.virtCam) virtLines.draw(V, P, opts.w, opts.h, 2 * dpr);
    gl.depthMask(false);
    grid.draw(V, P, camPos, sc.ground_seen ? 1 : 0.55);
    gl.depthMask(true);
  }

  function updateVirtFrustum() {
    const b = vBasis();
    const p = vcam.pos;
    const fpx = vcam.f35 * Math.hypot(W, H) / DIAG35;
    const z = clamp(V3.len(V3.sub(p, eye)) * 0.12, 0.1, 0.45);
    const corner = (u, v) => V3.add(p, V3.add(V3.scale(b.fwd, z), V3.add(V3.scale(b.right, (u - W / 2) / fpx * z), V3.scale(b.up, -(v - H / 2) / fpx * z))));
    const q = [corner(0, 0), corner(W, 0), corner(W, H), corner(0, H)];
    const col = [0.37, 0.88, 0.82, 1], segs = [];
    for (const k of q) segs.push({ a: p, b: k, c: col });
    for (let i = 0; i < 4; i++) segs.push({ a: q[i], b: q[(i + 1) % 4], c: col });
    const top = V3.scale(V3.add(q[0], q[1]), 0.5); // "up" tick so roll is visible
    segs.push({ a: top, b: V3.add(top, V3.scale(b.up, z * 0.25)), c: col });
    segs.push({ a: p, b: eye, c: [0.37, 0.88, 0.82, 0.55], dash: 0.06 });
    segs.push({ a: p, b: [p[0], 0, p[2]], c: [0.37, 0.88, 0.82, 0.45], dash: 0.05 });
    virtLines.set(segs);
  }

  function draw() {
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, vw, vh);
    gl.clearColor(0.043, 0.051, 0.07, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    updateVirtFrustum();
    let V, P, pos;
    if (mode === 'orbit') {
      const b = orbitBasis(); pos = orbitPos();
      V = M4.view(pos, b.right, b.up, b.back); P = lensProj(24, vw, vh);
    } else if (mode === 'camera') {
      const b = vBasis(); pos = vcam.pos;
      V = M4.view(pos, b.right, b.up, b.back); P = virtProj(vw, vh);
    } else {
      pos = photoCam.pos;
      V = M4.view(pos, photoCam.right, photoCam.up, photoCam.back);
      const s = Math.min(vw / W, vh / H);
      P = M4.fromIntrinsics(f * s, f * s, vw / 2, vh / 2, vw, vh);
    }
    // from the virtual camera, hide the original camera when it would sit right in front of the lens
    const nearPhotoCam = V3.len(V3.sub(vcam.pos, photoCam.pos)) < planeZ * 2.5;
    drawScene(V, P, pos, { w: vw, h: vh, photoCam: mode === 'orbit' || (mode === 'camera' && !nearPhotoCam), virtCam: mode === 'orbit' });

    if (mode === 'photo') {
      const s = Math.min(vw / W, vh / H);
      const cssW = W * s / dpr, cssH = H * s / dpr;
      Object.assign(overlay.style, { display: 'block', width: `${cssW}px`, height: `${cssH}px`, left: `calc(50% - ${cssW / 2}px)`, top: `calc(50% - ${cssH / 2}px)`, opacity: ui.opacity.value });
    } else overlay.style.display = 'none';

    // picture-in-picture: Overview shows what the virtual camera sees; Camera view shows the photo
    const narrow = vw / dpr < 760;
    const pipH = Math.round(Math.min(vh * (narrow ? 0.22 : 0.34), (vw * (narrow ? 0.4 : 0.26)) / aspect));
    const pipW = Math.round(pipH * aspect);
    const pipX = Math.round(12 * dpr);
    const pipTopCss = 64; // below the toolbar on narrow screens
    if (mode === 'photo') ui.pip.hidden = true;
    else {
      ui.pip.hidden = false;
      Object.assign(ui.pip.style, { width: `${pipW / dpr}px`, height: `${pipH / dpr}px`, left: '12px', bottom: narrow ? 'auto' : '12px', top: narrow ? `${pipTopCss}px` : 'auto' });
      if (mode === 'orbit') {
        ui.pip.className = 'v-pip'; ui.pipLabel.textContent = 'Virtual camera sees'; pipPhoto.style.display = 'none';
        const glY = narrow ? vh - Math.round(pipTopCss * dpr) - pipH : Math.round(12 * dpr);
        gl.enable(gl.SCISSOR_TEST);
        gl.viewport(pipX, glY, pipW, pipH);
        gl.scissor(pipX, glY, pipW, pipH);
        gl.clearColor(0.07, 0.085, 0.11, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        const b = vBasis();
        const fpx = vcam.f35 * Math.hypot(pipW, pipH) / DIAG35;
        drawScene(M4.view(vcam.pos, b.right, b.up, b.back), M4.fromIntrinsics(fpx, fpx, pipW / 2, pipH / 2, pipW, pipH), vcam.pos, { w: pipW, h: pipH, photoCam: !nearPhotoCam, virtCam: false });
        gl.disable(gl.SCISSOR_TEST);
      } else {
        ui.pip.className = 'v-pip photo'; ui.pipLabel.textContent = 'Original photo'; pipPhoto.style.display = 'block';
      }
    }
    updateLabels(V, P);
    updatePanel();
  }

  function updateLabels(V, P) {
    const VP = M4.mul(P, V);
    const defs = mode === 'photo' ? [] : [
      { key: 'cam', at: photoCam.pos, text: '📷 Original camera', click: () => setMode('photo') },
      { key: 'eye', at: eye, text: 'Eyes' },
      { key: 'd', at: V3.lerp(camFoot, eyeFoot, 0.5), text: `${c.horizontal_distance_m.v.toFixed(2)} m` },
      ...(mode === 'orbit' ? [{ key: 'vc', at: vcam.pos, text: '🎥 Virtual camera', click: () => setMode('camera') }] : []),
    ];
    const used = new Set(defs.map((d) => d.key));
    for (const k in labelEls) if (!used.has(k)) labelEls[k].style.display = 'none';
    for (const d of defs) {
      const e = mkLabel(d.key, d.text, d.click);
      const p = d.at;
      const x = VP[0] * p[0] + VP[4] * p[1] + VP[8] * p[2] + VP[12];
      const y = VP[1] * p[0] + VP[5] * p[1] + VP[9] * p[2] + VP[13];
      const w = VP[3] * p[0] + VP[7] * p[1] + VP[11] * p[2] + VP[15];
      if (w <= 0.05) { e.style.display = 'none'; continue; }
      const sx = (x / w * 0.5 + 0.5) * (vw / dpr), sy = (1 - (y / w * 0.5 + 0.5)) * (vh / dpr);
      if (sx < -50 || sy < -20 || sx > vw / dpr + 50 || sy > vh / dpr + 40) { e.style.display = 'none'; continue; }
      e.style.display = 'block';
      e.style.left = `${sx}px`; e.style.top = `${sy - 8}px`;
    }
  }

  // ---------- panel: sliders <-> rig ----------
  let rollSign = 1;
  {
    const save = vcam.roll; vcam.roll = 0.2;
    rollSign = Math.sign(metricsOf(vcam.pos, vBasis()).roll - metricsOf(vcam.pos, basisFrom(vcam.yaw, vcam.pitch, 0)).roll) || 1;
    vcam.roll = save;
  }
  const fmt = { az: (v) => `${v.toFixed(0)}°`, dist: (v) => `${v.toFixed(2)} m`, h: (v) => `${v.toFixed(2)} m`, tilt: (v) => `${v.toFixed(1)}°`, roll: (v) => `${v.toFixed(1)}°` };
  let editing = null; // slider being dragged: don't overwrite its value
  function updatePanel() {
    const m = metricsOf(vcam.pos, vBasis());
    const vals = { az: m.az, dist: m.dh, h: vcam.pos[1], tilt: m.pitch, roll: m.roll };
    for (const k in vals) {
      if (editing !== k) ui.s[k].value = String(vals[k]);
      ui.o[k].textContent = fmt[k](vals[k]);
    }
    ui.s.tilt.disabled = vcam.aim;
    const row = (k, v, ref, unit, d = 2) => {
      const delta = v - ref;
      const small = Math.abs(delta) < (unit === '°' ? 0.5 : 0.01);
      const ds = small ? '' : ` <span class="delta">${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(unit === '°' ? 0 : d)}</span>`;
      return `<tr><td>${k}</td><td>${v.toFixed(unit === '°' ? 1 : d)}${unit}${ds}</td></tr>`;
    };
    ui.hud.innerHTML = `<table>
      ${row('Height vs. eyes', m.hEye, c.height_above_eye_m.v, ' m')}
      ${row('Distance to eyes', m.dist, c.distance_to_eyes_m.v, ' m')}
      ${row('Tilt (+ down)', m.pitch, c.pitch_deg.v, '°')}
      ${row('Elevation', m.elev, c.elevation_deg.v, '°')}
      ${row('Side angle', m.az, c.azimuth_deg.v, '°')}
      <tr><td>Lens</td><td>${vcam.f35.toFixed(0)} mm</td></tr></table>
      <div class="delta" style="margin-top:4px">Grey numbers: change from the original shot.</div>`;
  }
  function onSlider(e) {
    const k = Object.keys(ui.s).find((key) => ui.s[key] === e.target);
    if (!k) return;
    editing = k;
    const v = +e.target.value;
    const m = metricsOf(vcam.pos, vBasis());
    if (k === 'az' || k === 'dist') {
      let az = (k === 'az' ? v : m.az) / DEG;
      if (photo.mirrored) az = -az;
      const dh = Math.max(0.05, k === 'dist' ? v : m.dh);
      const oldYawToEye = anglesFrom(V3.norm(V3.sub(eye, vcam.pos))).yaw;
      vcam.pos = [eye[0] + Math.sin(az) * dh, vcam.pos[1], eye[2] + Math.cos(az) * dh];
      if (vcam.aim) aimAtEyes();
      else vcam.yaw += anglesFrom(V3.norm(V3.sub(eye, vcam.pos))).yaw - oldYawToEye; // turn with the move
    } else if (k === 'h') {
      vcam.pos = [vcam.pos[0], v, vcam.pos[2]];
      if (vcam.aim) aimAtEyes();
    } else if (k === 'tilt') {
      vcam.pitch = -v / DEG;
    } else if (k === 'roll') {
      vcam.roll = rollSign * v / DEG;
    }
    requestRender();
  }
  const onSliderEnd = () => { editing = null; requestRender(); };
  Object.values(ui.s).forEach((s) => { s.addEventListener('input', onSlider); s.addEventListener('change', onSliderEnd); });
  const onAim = () => { vcam.aim = ui.aim.checked; if (vcam.aim) aimAtEyes(); requestRender(); };
  ui.aim.addEventListener('change', onAim);

  function syncLens() {
    ui.lens.querySelectorAll('button').forEach((b) => b.classList.toggle('on', Math.abs(+b.dataset.lens - vcam.f35) < 0.6));
  }
  function setLens(f35) {
    const from = { pos: vcam.pos.slice(), f35: vcam.f35 };
    let toPos = vcam.pos.slice();
    const dolly = ui.dolly.checked;
    if (dolly) toPos = V3.add(eye, V3.scale(V3.sub(vcam.pos, eye), f35 / vcam.f35));
    anim = { t0: performance.now(), dur: dolly ? 900 : 200, from, to: { pos: toPos, f35 }, aim: dolly };
    ui.lens.querySelectorAll('button').forEach((b) => b.classList.toggle('on', +b.dataset.lens === f35));
    requestRender();
  }

  function setMode(m) {
    mode = m;
    root.querySelectorAll('[data-mode]').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
    ui.opacityWrap.hidden = m !== 'photo';
    ui.help.innerHTML = m === 'orbit'
      ? 'Drag to rotate · right-drag or Shift-drag to pan · wheel to zoom · <b>W A S D / Q E</b> move the virtual camera'
      : m === 'camera' ? 'You are the virtual camera: drag to look · <b>W A S D</b> move · <b>Q / E</b> down / up · Shift faster · wheel forward/back'
        : 'The solved camera with the photo on top: use the slider to compare. Drag in the scene to take over as the virtual camera.';
    requestRender();
  }

  // ---------- input ----------
  function stepKeys(dt) {
    const speed = 1.4 * (keys.has('shift') ? 3 : 1) * dt;
    const b = vBasis();
    const fwd = V3.norm([b.fwd[0], 0, b.fwd[2]]);
    const right = V3.norm([b.right[0], 0, b.right[2]]);
    let mv = [0, 0, 0];
    if (keys.has('w') || keys.has('arrowup')) mv = V3.add(mv, fwd);
    if (keys.has('s') || keys.has('arrowdown')) mv = V3.sub(mv, fwd);
    if (keys.has('d') || keys.has('arrowright')) mv = V3.add(mv, right);
    if (keys.has('a') || keys.has('arrowleft')) mv = V3.sub(mv, right);
    if (keys.has('e') || keys.has(' ')) mv = V3.add(mv, [0, 1, 0]);
    if (keys.has('q') || keys.has('c')) mv = V3.sub(mv, [0, 1, 0]);
    if (V3.len(mv) < 1e-6) return false;
    if (mode === 'photo') setMode('camera');
    vcam.pos = V3.add(vcam.pos, V3.scale(V3.norm(mv), speed));
    vcam.pos[1] = Math.max(0.02, vcam.pos[1]);
    if (vcam.aim) aimAtEyes();
    return true;
  }

  const pointers = new Map();
  let drag = null, pinch = null;
  const onDown = (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    drag = { x: e.clientX, y: e.clientY, btn: e.button, shift: e.shiftKey, id: e.pointerId };
    if (pointers.size === 2) pinch = null;
    canvas.focus();
  };
  const onMove = (e) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch != null) zoomBy((pinch - d) * 4);
      pinch = d;
      return;
    }
    if (!drag || drag.id !== e.pointerId) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (mode === 'photo') setMode('camera');
    if (mode === 'orbit') {
      if (drag.btn === 2 || drag.shift) {
        const b = orbitBasis();
        const k = orbit.dist * 0.0016;
        orbit.target = V3.add(orbit.target, V3.add(V3.scale(b.right, -dx * k), V3.scale(b.up, dy * k)));
      } else {
        orbit.yaw -= dx * 0.007;
        orbit.pitch = clamp(orbit.pitch + dy * 0.006, -0.2, 1.5);
      }
    } else {
      if (vcam.aim) { vcam.aim = false; ui.aim.checked = false; } // looking around turns off auto-aim
      vcam.yaw += dx * 0.004;
      vcam.pitch = clamp(vcam.pitch + dy * 0.004, -1.55, 1.55);
    }
    requestRender();
  };
  const onUp = (e) => { pointers.delete(e.pointerId); if (drag && drag.id === e.pointerId) drag = null; if (pointers.size < 2) pinch = null; };
  const zoomBy = (delta) => {
    if (mode === 'orbit') orbit.dist = clamp(orbit.dist * Math.exp(delta * 0.0012), 0.5, 40);
    else {
      if (mode === 'photo') setMode('camera');
      vcam.pos = V3.add(vcam.pos, V3.scale(vBasis().fwd, -delta * 0.002));
      if (vcam.aim) aimAtEyes();
    }
    requestRender();
  };
  const onWheel = (e) => { e.preventDefault(); zoomBy(e.deltaY * (e.deltaMode === 1 ? 30 : 1)); };
  const keyName = (e) => (e.key === ' ' ? ' ' : e.key.toLowerCase());
  const MOVE = new Set(['w', 'a', 's', 'd', 'q', 'e', 'c', ' ', 'shift', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']);
  const onKeyDown = (e) => {
    if (e.target.matches?.('input[type=number], input[type=text], select, textarea')) return;
    const k = keyName(e);
    if (k === 'escape') { closeViewer(); return; }
    if (k === '1') setMode('orbit'); else if (k === '2') setMode('camera'); else if (k === '3') setMode('photo');
    if (!MOVE.has(k)) return;
    // arrow keys on a focused slider adjust the slider instead
    if (k.startsWith('arrow') && e.target.matches?.('input[type=range]')) return;
    e.preventDefault();
    keys.add(k);
    last = performance.now();
    requestRender();
  };
  const onKeyUp = (e) => keys.delete(keyName(e));
  const onBlur = () => keys.clear();

  canvas.tabIndex = 0;
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', onBlur);
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);

  const onClick = (e) => {
    const b = e.target.closest('button');
    if (!b || !root.contains(b)) return;
    if (b.dataset.mode) setMode(b.dataset.mode);
    if (b.dataset.lens) setLens(+b.dataset.lens);
    if (b.dataset.layer) { layers[b.dataset.layer] = !layers[b.dataset.layer]; b.classList.toggle('on', layers[b.dataset.layer]); requestRender(); }
    if (b.id === 'vReset') { anim = null; resetToPhoto(); syncLens(); requestRender(); }
    if (b.id === 'vLook') setMode('camera');
    if (b.id === 'vCollapse') { const col = ui.panel.classList.toggle('collapsed'); b.setAttribute('aria-expanded', String(!col)); }
    if (b.id === 'vClose') closeViewer();
  };
  root.addEventListener('click', onClick);
  const onOpacity = () => requestRender();
  ui.opacity.addEventListener('input', onOpacity);
  root.querySelector('[data-layer="cutout"]').hidden = !cutout;
  root.querySelectorAll('[data-layer]').forEach((b) => b.classList.toggle('on', !!layers[b.dataset.layer]));
  syncLens();

  state = {
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
      root.removeEventListener('click', onClick);
      ui.opacity.removeEventListener('input', onOpacity);
      ui.aim.removeEventListener('change', onAim);
      Object.values(ui.s).forEach((s) => { s.removeEventListener('input', onSlider); s.removeEventListener('change', onSliderEnd); });
      const fresh = canvas.cloneNode(false);
      canvas.replaceWith(fresh);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      ui.labels.innerHTML = '';
      overlay.style.display = 'none';
      root.hidden = true;
      onClose?.();
    },
    debug: {
      vcam, orbit, photoCam, metricsOf, vBasis, setMode, toW, ray, W, H, f,
      get mode() { return mode; }, get vw() { return vw; }, get vh() { return vh; },
      photoProjection: () => { const s = Math.min(vw / W, vh / H); return { P: M4.fromIntrinsics(f * s, f * s, vw / 2, vh / 2, vw, vh), s }; },
    },
  };
  window.__camposeViewer = state.debug;
  setMode('orbit');
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
