// Minimal WebGL2 helpers: column-major mat4, shader programs, the 4 renderers we need.

export const M4 = {
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return o;
  },
  /** View matrix from camera position and basis (right, up, back) in world. */
  view(pos, right, up, back) {
    const d = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return new Float32Array([
      right[0], up[0], back[0], 0,
      right[1], up[1], back[1], 0,
      right[2], up[2], back[2], 0,
      -d(right, pos), -d(up, pos), -d(back, pos), 1,
    ]);
  },
  /**
   * OpenGL projection from pinhole intrinsics (pixels) for a W×H viewport.
   * Matches the plan's P with the principal point at (cx, cy) in top-left pixel coords.
   */
  fromIntrinsics(fx, fy, cx, cy, W, H, n = 0.02, f = 200) {
    return new Float32Array([
      2 * fx / W, 0, 0, 0,
      0, 2 * fy / H, 0, 0,
      1 - 2 * cx / W, 2 * cy / H - 1, -(f + n) / (f - n), -1,
      0, 0, -2 * f * n / (f - n), 0,
    ]);
  },
};

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
export function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}

// ---- thick screen-space lines (instanced quads, near-plane clipped, optional dashes)
const LINE_VS = `#version 300 es
layout(location=0) in vec2 aQuad;
layout(location=1) in vec3 aA;
layout(location=2) in vec3 aB;
layout(location=3) in vec4 aColor;
layout(location=4) in float aDash;
uniform mat4 uV, uP;
uniform vec2 uViewport;
uniform float uWidth;
out vec4 vColor;
out float vAlong;
out float vDash;
void main() {
  vec4 va = uV * vec4(aA, 1.0), vb = uV * vec4(aB, 1.0);
  float nearZ = -0.03;
  float len = length(aB - aA);
  float t0 = 0.0, t1 = 1.0;
  if (va.z > nearZ && vb.z > nearZ) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  if (va.z > nearZ) { float t = (nearZ - va.z) / (vb.z - va.z); va = mix(va, vb, t); t0 = t; }
  if (vb.z > nearZ) { float t = (nearZ - va.z) / (vb.z - va.z); vb = mix(va, vb, t); t1 = t0 + (1.0 - t0) * t; }
  vec4 ca = uP * va, cb = uP * vb;
  vec2 sa = ca.xy / ca.w * uViewport, sb = cb.xy / cb.w * uViewport;
  vec2 dir = normalize(sb - sa + 1e-6);
  vec2 nrm = vec2(-dir.y, dir.x);
  vec4 c = aQuad.x < 0.5 ? ca : cb;
  c.xy += nrm * aQuad.y * uWidth / uViewport * c.w;
  gl_Position = c;
  vColor = aColor;
  vAlong = mix(t0, t1, aQuad.x) * len;
  vDash = aDash;
}`;
const LINE_FS = `#version 300 es
precision mediump float;
in vec4 vColor; in float vAlong; in float vDash;
out vec4 o;
void main() {
  if (vDash > 0.0 && fract(vAlong / vDash) > 0.55) discard;
  o = vColor;
}`;

export class Lines {
  constructor(gl) {
    this.gl = gl;
    this.prog = program(gl, LINE_VS, LINE_FS);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const q = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, q);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    const stride = 11 * 4;
    [[1, 3, 0], [2, 3, 12], [3, 4, 24], [4, 1, 40]].forEach(([loc, size, off]) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, off);
      gl.vertexAttribDivisor(loc, 1);
    });
    gl.bindVertexArray(null);
    this.n = 0;
  }
  /** segs: array of { a:[x,y,z], b:[x,y,z], c:[r,g,b,a], dash?:meters } */
  set(segs) {
    const d = new Float32Array(segs.length * 11);
    segs.forEach((s, i) => { d.set([...s.a, ...s.b, ...s.c, s.dash || 0], i * 11); });
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, d, gl.STATIC_DRAW);
    this.n = segs.length;
  }
  draw(V, P, vw, vh, width) {
    if (!this.n) return;
    const gl = this.gl, u = this.prog.u;
    gl.useProgram(this.prog.p);
    gl.uniformMatrix4fv(u.uV, false, V);
    gl.uniformMatrix4fv(u.uP, false, P);
    gl.uniform2f(u.uViewport, vw / 2, vh / 2);
    gl.uniform1f(u.uWidth, width);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.n);
    gl.bindVertexArray(null);
  }
}

// ---- round point sprites
const PT_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
uniform mat4 uV, uP;
uniform float uSize;
out vec4 vColor;
void main() {
  vec4 v = uV * vec4(aPos, 1.0);
  gl_Position = uP * v;
  // size in pixels, mildly attenuated with distance
  gl_PointSize = clamp(uSize * 2.0 / max(-v.z, 0.2), 1.5, uSize * 3.0);
  vColor = aColor;
}`;
const PT_FS = `#version 300 es
precision mediump float;
in vec4 vColor; out vec4 o;
void main() { vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard; o = vColor; }`;

export class Points {
  constructor(gl) {
    this.gl = gl;
    this.prog = program(gl, PT_VS, PT_FS);
    this.vao = gl.createVertexArray();
    this.buf = gl.createBuffer();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
    gl.bindVertexArray(null);
    this.n = 0;
  }
  set(pos, colors) {
    const n = pos.length / 3;
    const d = new Float32Array(n * 7);
    for (let i = 0; i < n; i++) { d.set(pos.subarray ? pos.subarray(i * 3, i * 3 + 3) : pos.slice(i * 3, i * 3 + 3), i * 7); d.set(colors.slice(i * 4, i * 4 + 4), i * 7 + 3); }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, d, gl.STATIC_DRAW);
    this.n = n;
  }
  draw(V, P, size) {
    if (!this.n) return;
    const gl = this.gl, u = this.prog.u;
    gl.useProgram(this.prog.p);
    gl.uniformMatrix4fv(u.uV, false, V);
    gl.uniformMatrix4fv(u.uP, false, P);
    gl.uniform1f(u.uSize, size);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.n);
    gl.bindVertexArray(null);
  }
}

// ---- textured quads (photo plane, cut-out billboard)
const TQ_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
uniform mat4 uV, uP;
out vec2 vUV;
void main() { gl_Position = uP * uV * vec4(aPos, 1.0); vUV = aUV; }`;
const TQ_FS = `#version 300 es
precision mediump float;
in vec2 vUV; uniform sampler2D uTex; uniform float uAlphaTest; uniform float uOpacity;
out vec4 o;
void main() { vec4 c = texture(uTex, vUV); if (c.a < uAlphaTest) discard; o = vec4(c.rgb, uOpacity); }`;

export class TexQuad {
  constructor(gl, source, corners) {
    this.gl = gl;
    if (!TexQuad.prog) TexQuad.prog = new WeakMap();
    let pr = TexQuad.prog.get(gl);
    if (!pr) { pr = program(gl, TQ_VS, TQ_FS); TexQuad.prog.set(gl, pr); }
    this.prog = pr;
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    // corners: top-left, top-right, bottom-left, bottom-right (image order)
    const [tl, tr, bl, br] = corners;
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([...tl, 0, 0, ...tr, 1, 0, ...bl, 0, 1, ...br, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
    gl.bindVertexArray(null);
  }
  draw(V, P, alphaTest = 0, opacity = 1) {
    const gl = this.gl, u = this.prog.u;
    gl.useProgram(this.prog.p);
    gl.uniformMatrix4fv(u.uV, false, V);
    gl.uniformMatrix4fv(u.uP, false, P);
    gl.uniform1f(u.uAlphaTest, alphaTest);
    gl.uniform1f(u.uOpacity, opacity);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(u.uTex, 0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }
}

// ---- procedural ground grid (one quad, anti-aliased lines, fades with distance)
const GRID_VS = `#version 300 es
layout(location=0) in vec2 aXZ;
uniform mat4 uV, uP; uniform float uR;
out vec3 vW;
void main() { vW = vec3(aXZ.x * uR, 0.0, aXZ.y * uR); gl_Position = uP * uV * vec4(vW, 1.0); }`;
const GRID_FS = `#version 300 es
precision highp float;
in vec3 vW; uniform vec3 uCam; uniform float uStrength;
out vec4 o;
float grid(vec2 p, float s) {
  vec2 g = abs(fract(p / s - 0.5) - 0.5) / fwidth(p / s);
  return 1.0 - min(min(g.x, g.y), 1.0);
}
void main() {
  vec2 p = vW.xz;
  float minor = grid(p, 0.5) * 0.35;
  float major = grid(p, 1.0);
  float axis = 0.0;
  vec2 fw = fwidth(p);
  if (abs(p.x) < fw.x * 1.5) axis = 1.0;
  if (abs(p.y) < fw.y * 1.5) axis = 1.0;
  float d = length(vW - uCam);
  float fade = clamp(1.0 - length(p) / 14.0, 0.0, 1.0) * clamp(1.4 - d / 30.0, 0.0, 1.0);
  float a = max(max(minor, major * 0.7), axis * 0.9) * fade * uStrength;
  vec3 col = mix(vec3(0.55, 0.62, 0.75), vec3(0.35, 0.65, 1.0), axis);
  // soft spot under the subject
  float spot = exp(-dot(p, p) * 1.8) * 0.18 * uStrength;
  o = vec4(col, max(a, spot));
  if (o.a < 0.01) discard;
}`;

export class Grid {
  constructor(gl) {
    this.gl = gl;
    this.prog = program(gl, GRID_VS, GRID_FS);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }
  draw(V, P, cam, strength) {
    const gl = this.gl, u = this.prog.u;
    gl.useProgram(this.prog.p);
    gl.uniformMatrix4fv(u.uV, false, V);
    gl.uniformMatrix4fv(u.uP, false, P);
    gl.uniform1f(u.uR, 16);
    gl.uniform3f(u.uCam, cam[0], cam[1], cam[2]);
    gl.uniform1f(u.uStrength, strength);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }
}

// ---- lit "clay" meshes (mannequin, camera bodies)
const MESH_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aN;
layout(location=2) in vec3 aCol;
uniform mat4 uV, uP;
out vec3 vN; out vec3 vCol; out vec3 vViewN;
void main() {
  gl_Position = uP * uV * vec4(aPos, 1.0);
  vN = aN; vCol = aCol;
  vViewN = mat3(uV) * aN;
}`;
const MESH_FS = `#version 300 es
precision mediump float;
in vec3 vN; in vec3 vCol; in vec3 vViewN;
out vec4 o;
void main() {
  vec3 n = normalize(vN);
  vec3 key = normalize(vec3(0.45, 0.85, 0.35));
  float d = max(dot(n, key), 0.0);
  float sky = 0.5 + 0.5 * n.y;                       // hemisphere ambient
  float head = max(dot(normalize(vViewN), vec3(0.0, 0.0, 1.0)), 0.0); // soft headlight
  float rim = pow(1.0 - head, 3.0) * 0.25;
  vec3 c = vCol * (0.22 + 0.25 * sky + 0.45 * d + 0.25 * head) + rim;
  o = vec4(c, 1.0);
}`;

export class Mesh {
  constructor(gl) {
    this.gl = gl;
    this.prog = program(gl, MESH_VS, MESH_FS);
    this.vao = gl.createVertexArray();
    this.bufs = [gl.createBuffer(), gl.createBuffer(), gl.createBuffer()];
    gl.bindVertexArray(this.vao);
    this.bufs.forEach((b, i) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 0, 0);
    });
    gl.bindVertexArray(null);
    this.n = 0;
  }
  set({ pos, nrm, col }) {
    const gl = this.gl;
    [pos, nrm, col].forEach((d, i) => { gl.bindBuffer(gl.ARRAY_BUFFER, this.bufs[i]); gl.bufferData(gl.ARRAY_BUFFER, d, gl.STATIC_DRAW); });
    this.n = pos.length / 3;
  }
  draw(V, P) {
    if (!this.n) return;
    const gl = this.gl, u = this.prog.u;
    gl.useProgram(this.prog.p);
    gl.uniformMatrix4fv(u.uV, false, V);
    gl.uniformMatrix4fv(u.uP, false, P);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, this.n);
    gl.bindVertexArray(null);
  }
}
