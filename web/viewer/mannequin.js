// Plain "clay" mannequin built from the solved 3D joints (and face points for the head).
// No detail, just the right shape and orientation, so any view of it reflects the real pose.

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const mid = (a, b) => mul(add(a, b), 0.5);
const lerp = (a, b, t) => add(a, mul(sub(b, a), t));
function perp(d) {
  const p = Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const x = norm(cross(d, p));
  return [x, cross(d, x)];
}

const CLAY = [0.80, 0.82, 0.86];
const LEFT = [0.93, 0.74, 0.55];  // subject's left: warm
const RIGHT = [0.58, 0.72, 0.92]; // subject's right: cool
const DARK = [0.16, 0.17, 0.2];

class Builder {
  constructor() { this.p = []; this.n = []; this.c = []; }
  tri(a, b, c, na, nb, nc, col) {
    this.p.push(...a, ...b, ...c);
    this.n.push(...na, ...nb, ...nc);
    for (let i = 0; i < 3; i++) this.c.push(...col);
  }
  /** Ellipsoid with semi-axes along unit vectors ax, ay, az. */
  ellipsoid(c, ax, ay, az, rx, ry, rz, col, lat = 8, lon = 14) {
    const pt = (i, j) => {
      const th = Math.PI * i / lat, ph = 2 * Math.PI * j / lon;
      const u = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
      const p = add(c, add(add(mul(ax, u[0] * rx), mul(ay, u[1] * ry)), mul(az, u[2] * rz)));
      const n = norm(add(add(mul(ax, u[0] / rx), mul(ay, u[1] / ry)), mul(az, u[2] / rz)));
      return [p, n];
    };
    for (let i = 0; i < lat; i++) for (let j = 0; j < lon; j++) {
      const [a, na] = pt(i, j), [b, nb] = pt(i + 1, j), [d, nd] = pt(i, j + 1), [e, ne] = pt(i + 1, j + 1);
      this.tri(a, b, e, na, nb, ne, col);
      this.tri(a, e, d, na, ne, nd, col);
    }
  }
  sphere(c, r, col) { this.ellipsoid(c, [1, 0, 0], [0, 1, 0], [0, 0, 1], r, r, r, col, 6, 10); }
  /** Tapered cylinder from a (radius ra) to b (radius rb), with round joints. */
  limb(a, b, ra, rb, col, seg = 10) {
    const d = sub(b, a);
    const L = len(d);
    if (L < 1e-4) return;
    const dn = mul(d, 1 / L);
    const [x, y] = perp(dn);
    const ring = (c, r, k) => {
      const ang = 2 * Math.PI * k / seg;
      const n = add(mul(x, Math.cos(ang)), mul(y, Math.sin(ang)));
      return [add(c, mul(n, r)), n];
    };
    for (let k = 0; k < seg; k++) {
      const [p0, n0] = ring(a, ra, k), [p1, n1] = ring(a, ra, k + 1);
      const [q0, m0] = ring(b, rb, k), [q1, m1] = ring(b, rb, k + 1);
      this.tri(p0, q0, q1, n0, m0, m1, col);
      this.tri(p0, q1, p1, n0, m1, n1, col);
    }
    this.sphere(a, ra, col);
    this.sphere(b, rb, col);
  }
  /** Lofted torso through elliptical rings (centre, half-width, half-depth, axes). */
  loft(rings, col, seg = 16) {
    const pt = (r, k) => {
      const ang = 2 * Math.PI * k / seg;
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const p = add(r.c, add(mul(r.x, ca * r.w), mul(r.z, sa * r.d)));
      const n = norm(add(mul(r.x, ca / r.w), mul(r.z, sa / r.d)));
      return [p, n];
    };
    for (let i = 0; i + 1 < rings.length; i++) for (let k = 0; k < seg; k++) {
      const [a, na] = pt(rings[i], k), [b, nb] = pt(rings[i], k + 1);
      const [c, nc] = pt(rings[i + 1], k), [d, nd] = pt(rings[i + 1], k + 1);
      this.tri(a, c, d, na, nc, nd, col);
      this.tri(a, d, b, na, nd, nb, col);
    }
    // caps
    for (const [r, s] of [[rings[0], -1], [rings[rings.length - 1], 1]]) {
      const up = mul(r.y, s);
      for (let k = 0; k < seg; k++) {
        const [a] = pt(r, k), [b] = pt(r, k + 1);
        if (s > 0) this.tri(r.c, a, b, up, up, up, col); else this.tri(r.c, b, a, up, up, up, col);
      }
    }
  }
  out() { return { pos: new Float32Array(this.p), nrm: new Float32Array(this.n), col: new Float32Array(this.c) }; }
}

/** Head pose (centre, axes, size) from face points if present, else from body landmarks. */
function headPose(J, face) {
  if (face) {
    const F = (i) => [face[i * 3], face[i * 3 + 1], face[i * 3 + 2]];
    const X = norm(sub(F(263), F(33)));           // subject's left
    let Y = sub(F(10), F(152));                    // chin -> forehead
    const faceH = len(Y);
    Y = norm(sub(Y, mul(X, dot(Y, X))));
    const Z = cross(X, Y);                         // face forward
    const eyes = mid(F(468), F(473));
    const s = Math.min(1.3, Math.max(0.75, faceH / 0.19));
    const c = add(add(eyes, mul(Z, -0.075 * s)), mul(Y, 0.015 * s));
    return { c, X, Y, Z, s };
  }
  if (!J) return null;
  const P = (i) => J(i);
  const X = norm(sub(P(7), P(8)));                 // left ear - right ear
  const earMid = mid(P(7), P(8));
  let Z = sub(P(0), earMid);                       // towards the nose
  Z = norm(sub(Z, mul(X, dot(Z, X))));
  const Y = cross(Z, X);
  const s = Math.min(1.3, Math.max(0.75, len(sub(P(7), P(8))) / 0.15));
  return { c: add(earMid, mul(Y, 0.01)), X, Y, Z, s };
}

/**
 * joints: flat [33*3] in subject frame W (or null), w8: per-joint confidence,
 * face: flat [478*3] in W (or null). Returns mesh arrays.
 */
export function buildMannequin(joints, w8, face) {
  const b = new Builder();
  const J = joints ? (i) => [joints[i * 3], joints[i * 3 + 1], joints[i * 3 + 2]] : null;
  const head = headPose(J, face);
  if (head) {
    const { c, X, Y, Z, s } = head;
    b.ellipsoid(c, X, Y, Z, 0.075 * s, 0.105 * s, 0.095 * s, CLAY);
    // nose and eyes make the facing direction obvious from any angle
    const nose = add(add(c, mul(Z, 0.1 * s)), mul(Y, -0.02 * s));
    b.ellipsoid(nose, X, Y, Z, 0.014 * s, 0.026 * s, 0.022 * s, CLAY, 5, 8);
    for (const side of [-1, 1]) {
      b.sphere(add(add(add(c, mul(Z, 0.083 * s)), mul(X, side * 0.032 * s)), mul(Y, 0.012 * s)), 0.011 * s, DARK);
    }
  }
  if (J) {
    const conf = (i) => (w8 && w8.length ? w8[i] : 1);
    const tone = (col, ...ids) => (Math.min(...ids.map(conf)) > 0.3 ? col : col.map((v) => v * 0.6 + 0.12));
    const shL = J(11), shR = J(12), hipL = J(23), hipR = J(24);
    const shM = mid(shL, shR), hipM = mid(hipL, hipR);
    const up = norm(sub(shM, hipM));
    const leftV = norm(add(sub(shL, shR), sub(hipL, hipR)));
    const fwd = norm(cross(leftV, up));
    const xAx = norm(cross(up, fwd));
    const shW = len(sub(shL, shR)) / 2, hipW = len(sub(hipL, hipR)) / 2;
    const ring = (t, wMul, d) => {
      const c = lerp(hipM, shM, t);
      return { c, x: xAx, y: up, z: fwd, w: Math.max(0.08, (hipW + (shW - hipW) * t) * wMul), d };
    };
    b.loft([ring(-0.08, 1.25, 0.095), ring(0.3, 1.05, 0.09), ring(0.62, 1.05, 0.105), ring(0.9, 0.92, 0.1), ring(1.02, 0.7, 0.07)], tone(CLAY, 11, 12, 23, 24));
    // neck to head
    if (head) b.limb(add(shM, mul(up, 0.02)), add(head.c, mul(head.Y, -0.06 * head.s)), 0.05, 0.045, CLAY, 8);
    const arm = (s, e, w, hIdx, col) => {
      b.limb(J(s), J(e), 0.048, 0.042, tone(col, s, e));
      b.limb(J(e), J(w), 0.04, 0.032, tone(col, e, w));
      b.sphere(mid(J(w), J(hIdx)), 0.042, tone(col, w));
    };
    arm(11, 13, 15, 19, LEFT);
    arm(12, 14, 16, 20, RIGHT);
    const leg = (h, k, a, heel, toe, col) => {
      b.limb(J(h), J(k), 0.075, 0.055, tone(col, h, k));
      b.limb(J(k), J(a), 0.052, 0.04, tone(col, k, a));
      b.limb(J(heel), J(toe), 0.035, 0.03, tone(col, heel, toe));
    };
    leg(23, 25, 27, 29, 31, LEFT);
    leg(24, 26, 28, 30, 32, RIGHT);
  } else if (head) {
    // face-only photo: a neck stub so the head doesn't float
    b.limb(add(head.c, mul(head.Y, -0.07 * head.s)), add(head.c, mul(head.Y, -0.2 * head.s)), 0.05, 0.055, CLAY, 8);
  }
  return b.out();
}
