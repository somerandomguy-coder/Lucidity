// Thin wrapper over the Rust geometry core (raw wasm exports, no wasm-bindgen).
let ex = null;
let mem = null;
let inBuf = null; // reusable f64 input buffer inside wasm memory
let inLen = 0;
const dec = new TextDecoder();

export async function initCore(url = new URL('./campose_core.wasm', import.meta.url)) {
  const bytes = await (await fetch(url)).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  ex = instance.exports;
  mem = ex.memory;
  inLen = ex.cp_input_len();
  inBuf = ex.cp_alloc(inLen * 8);
}

export function inputLen() { return inLen; }
/** Tuning: body rotation prior σ in degrees (default 3). */
export function setRotPrior(deg) { ex.cp_set_rot_prior(deg); }

function readOut(n) {
  return JSON.parse(dec.decode(new Uint8Array(mem.buffer, ex.cp_out_ptr(), n)));
}

/** Geometry only: <5 ms. `input` is a Float64Array(inputLen()). */
export function analyze(input) {
  new Float64Array(mem.buffer, inBuf, inLen).set(input);
  return readOut(ex.cp_analyze(inBuf, inLen));
}

/** Intrinsics from file header bytes (EXIF) + decoded size. */
export function intrinsics(headBytes, w, h, presetF35, forcePreset) {
  const n = headBytes ? headBytes.byteLength : 0;
  let p = 0;
  if (n) {
    p = ex.cp_alloc(n);
    new Uint8Array(mem.buffer, p, n).set(headBytes);
  }
  const out = readOut(ex.cp_intrinsics(p, n, w, h, presetF35, forcePreset ? 1 : 0));
  if (n) ex.cp_free(p, n);
  return out;
}

// Input buffer layout (must match crates/core/src/solve.rs)
export const HDR = 16, N_POSE = 33, N_FACE = 478;
export const OFF_POSE2D = HDR, OFF_WORLD = HDR + N_POSE * 3, OFF_FACE = OFF_WORLD + N_POSE * 3;
