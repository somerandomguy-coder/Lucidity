// Stage 2: landmark models (MediaPipe Tasks Vision, WASM + optional GPU) and
// per-photo preprocessing. See docs/decisions.md for why this replaces tract/ONNX.
import { HDR, N_POSE, N_FACE, OFF_POSE2D, OFF_WORLD, OFF_FACE, inputLen, intrinsics } from './core.js';

const TV_VERSION = '1.2.3';
const TV_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TV_VERSION}`;
const MODELS = {
  pose_lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
  pose_full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
  face: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
};
const CACHE = 'campose-models-v1';
/**
 * Longest side used for inference (every photo is scaled to exactly this, up or down).
 * Landmark models work on ≤256 px crops, so more is wasted work.
 */
export const ANALYSIS_MAX = 1024;

let vision = null;
let poseLm = null;
let faceLm = null;
let loadedKey = '';
export let delegateUsed = 'CPU';

async function fetchCached(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* no Cache Storage (insecure context) */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) { onProgress?.(1); return new Uint8Array(await hit.arrayBuffer()); }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed: ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress?.(got / total);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  if (cache) { try { await cache.put(url, new Response(buf)); } catch { /* quota */ } }
  return buf;
}

/** Load (or reload) models. settings: { model: 'lite'|'full', processor: 'auto'|'cpu' } */
export async function loadModels(settings, onProgress) {
  const key = `${settings.model}|${settings.processor}`;
  if (key === loadedKey && poseLm && faceLm) return;
  const mod = await import(`${TV_BASE}/vision_bundle.mjs`);
  if (!vision) vision = await mod.FilesetResolver.forVisionTasks(`${TV_BASE}/wasm`);
  const parts = [0, 0];
  const report = () => onProgress?.((parts[0] + parts[1]) / 2);
  const [poseBytes, faceBytes] = await Promise.all([
    fetchCached(settings.model === 'full' ? MODELS.pose_full : MODELS.pose_lite, (p) => { parts[0] = p; report(); }),
    fetchCached(MODELS.face, (p) => { parts[1] = p; report(); }),
  ]);
  poseLm?.close(); faceLm?.close();
  const make = async (delegate) => {
    const pose = await mod.PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetBuffer: poseBytes, delegate },
      runningMode: 'IMAGE', numPoses: 2,
      minPoseDetectionConfidence: 0.4, minPosePresenceConfidence: 0.4,
    });
    const face = await mod.FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetBuffer: faceBytes, delegate },
      runningMode: 'IMAGE', numFaces: 2,
      outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
      minFaceDetectionConfidence: 0.4,
    });
    return [pose, face];
  };
  let pair = null;
  if (settings.processor !== 'cpu') {
    try { pair = await make('GPU'); delegateUsed = 'GPU'; } catch (e) { console.warn('GPU delegate failed, using CPU', e); }
  }
  if (!pair) { pair = await make('CPU'); delegateUsed = 'CPU'; }
  [poseLm, faceLm] = pair;
  loadedKey = key;
}

function bboxArea(lms) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const p of lms) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  return (x1 - x0) * (y1 - y0);
}
function largest(list) {
  let best = -1, bi = -1;
  list.forEach((l, i) => { const a = bboxArea(l); if (a > best) { best = a; bi = i; } });
  return bi;
}

/**
 * Decode a file into a bitmap no larger than `maxSide` (EXIF orientation applied).
 * With `exact`, small images are scaled *up* too, so every copy of a photo reaches the
 * models at the same size and gives the same result whatever its file resolution.
 */
export async function decode(file, maxSide, exact = false) {
  const full = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const W0 = full.width, H0 = full.height;
  const sc = exact ? maxSide / Math.max(W0, H0) : Math.min(1, maxSide / Math.max(W0, H0));
  if (Math.abs(sc - 1) < 1e-3) return { bmp: full, W0, H0, sc: 1 };
  const bmp = await createImageBitmap(full, {
    resizeWidth: Math.round(W0 * sc), resizeHeight: Math.round(H0 * sc), resizeQuality: 'medium',
  });
  full.close();
  return { bmp, W0, H0, sc: bmp.width / W0 };
}

async function makeThumb(bmp, side = 360) {
  const sc = Math.min(1, side / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * sc), h = Math.round(bmp.height * sc);
  const c = new OffscreenCanvas(w, h);
  c.getContext('2d').drawImage(bmp, 0, 0, w, h);
  const blob = await c.convertToBlob({ type: 'image/jpeg', quality: 0.8 });
  return { url: URL.createObjectURL(blob), w, h };
}

/**
 * Run detection once per photo. Returns everything needed to re-solve geometry
 * later without touching the models again.
 */
export async function detectPhoto(file, settings) {
  const head = new Uint8Array(await file.slice(0, 256 * 1024).arrayBuffer());
  const t0 = performance.now();
  const { bmp, W0, H0 } = await decode(file, ANALYSIS_MAX, true);
  const w = bmp.width, h = bmp.height;
  const thumb = await makeThumb(bmp);
  const pr = poseLm.detect(bmp);
  const fr = faceLm.detect(bmp);
  bmp.close();
  const ms = performance.now() - t0;

  const input = new Float64Array(inputLen());
  input[0] = w; input[1] = h;
  let hasPose = false, hasFace = false;
  const pi = largest(pr.landmarks || []);
  if (pi >= 0) {
    hasPose = true;
    const L = pr.landmarks[pi], Wd = pr.worldLandmarks[pi];
    const visKnown = L.some((p) => (p.visibility ?? 0) > 0);
    for (let i = 0; i < N_POSE; i++) {
      const p = L[i];
      const inFrame = p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
      input[OFF_POSE2D + i * 3] = p.x * w;
      input[OFF_POSE2D + i * 3 + 1] = p.y * h;
      input[OFF_POSE2D + i * 3 + 2] = visKnown ? p.visibility : (inFrame ? 0.9 : 0.1);
      input[OFF_WORLD + i * 3] = Wd[i].x;
      input[OFF_WORLD + i * 3 + 1] = Wd[i].y;
      input[OFF_WORLD + i * 3 + 2] = Wd[i].z;
    }
  }
  const fi = largest(fr.faceLandmarks || []);
  if (fi >= 0 && fr.faceLandmarks[fi].length >= N_FACE) {
    hasFace = true;
    const F = fr.faceLandmarks[fi];
    for (let i = 0; i < N_FACE; i++) {
      input[OFF_FACE + i * 3] = F[i].x * w;
      input[OFF_FACE + i * 3 + 1] = F[i].y * h;
      input[OFF_FACE + i * 3 + 2] = F[i].z * w;
    }
  }
  input[8] = hasPose ? 1 : 0;
  input[9] = hasFace ? 1 : 0;
  return { input, head, W0, H0, w, h, thumb, ms, hasPose, hasFace, people: (pr.landmarks || []).length };
}

/**
 * Fill the assumption fields of the input buffer (re-solve without inference).
 * a: { lensMode: 'auto'|'preset', f35, subjectH|null, upright, mirrored|null, mc }
 */
export function applyAssumptions(photo, a) {
  const intr = intrinsics(photo.head, photo.W0, photo.H0, a.f35, a.lensMode === 'preset');
  const sc = photo.w / photo.W0;
  const inp = photo.input;
  inp[2] = intr.f_px * sc;
  inp[3] = intr.sigma_rel;
  inp[4] = a.subjectH || 0;
  inp[5] = a.upright ? 1 : 0;
  const mirrored = a.mirrored ?? intr.front_camera;
  inp[6] = mirrored ? 1 : 0;
  inp[7] = a.mc ?? 64;
  inp[10] = intr.source === 'preset' ? 1 : 0;
  inp[11] = intr.cropped ? 1 : 0;
  inp[12] = 12345;
  // no EXIF and no lens picked by the user: let the solver estimate focal length from the body
  inp[13] = a.lensMode === 'auto' && intr.source === 'preset' ? 1 : 0;
  return { intr, mirrored };
}

/**
 * For the 3D view only: decode the selected photo at a moderate size and cut the
 * person out with the pose model's segmentation mask (one extra inference).
 * Returns { photo: canvas, cutout: canvas|null, w, h }.
 */
export async function preparePhotoFor3D(file, maxSide = 1280) {
  const { bmp } = await decode(file, maxSide);
  const w = bmp.width, h = bmp.height;
  const photo = document.createElement('canvas');
  photo.width = w; photo.height = h;
  const ctx = photo.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  let cutout = null;
  try {
    poseLm.setOptions({ outputSegmentationMasks: true });
    const r = poseLm.detect(bmp);
    const masks = r.segmentationMasks || [];
    if (masks.length && r.landmarks?.length) {
      const mi = largest(r.landmarks);
      const m = masks[Math.min(mi, masks.length - 1)];
      const mw = m.width, mh = m.height;
      const data = m.getAsFloat32Array();
      // mask may be smaller than the image: sample it
      const img = ctx.getImageData(0, 0, w, h);
      const px = img.data;
      for (let y = 0; y < h; y++) {
        const my = Math.min(mh - 1, (y * mh / h) | 0);
        for (let x = 0; x < w; x++) {
          const mx = Math.min(mw - 1, (x * mw / w) | 0);
          const a = data[my * mw + mx];
          px[(y * w + x) * 4 + 3] = a > 0.5 ? 255 : Math.max(0, (a - 0.2) * 850) | 0;
        }
      }
      cutout = document.createElement('canvas');
      cutout.width = w; cutout.height = h;
      cutout.getContext('2d').putImageData(img, 0, 0);
    }
    masks.forEach((m) => m.close?.());
  } catch (e) {
    console.warn('segmentation failed', e);
  } finally {
    try { poseLm.setOptions({ outputSegmentationMasks: false }); } catch { /* ignore */ }
  }
  bmp.close();
  return { photo, cutout, w, h };
}

export { HDR };
