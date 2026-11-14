# CamPose Web — Camera Angle & Position from a Photo

Nov 14, 2026 · @Khang

## Overview and prior art

CamPose Web takes one photo of a person (face close-up or full body) and returns the camera's tilt, roll, field of view, distance to the subject, and height relative to the subject's eyes and feet. It then rebuilds the shot as a small 3D scene. Everything runs locally in the browser: a Rust core compiled to WebAssembly, small landmark models, and a hand-written WebGL2 viewer.

**Core idea.** A human body and face are a calibration object of roughly known size. With focal length from EXIF (or a lens preset) and 2D landmarks from a small network, a perspective-n-point (PnP) solve gives the camera pose relative to the person. Gravity comes from the subject's upright posture, or from scene lines when they exist. No heavy depth or calibration network is needed for the MVP.

**Who already does parts of this.** Nobody I found ships this combination: numeric camera pose relative to a person, computed in the browser, plus a 3D rebuild. The pieces exist separately:

| Work | What it does | Gap vs. CamPose |
| --- | --- | --- |
| [SPEC](https://spec.is.tue.mpg.de/) (ICCV 2021) | A network estimates field of view, pitch and roll from an image of people, then a perspective-aware body mesh is fit | PyTorch on GPU, research-only license, no app |
| [CameraHMR](https://arxiv.org/pdf/2411.08128) (2024) | HumanFoV network predicts vertical FOV from human images; perspective body mesh | Same: heavy, research code |
| [GeoCalib](https://github.com/cvg/GeoCalib) (ECCV 2024) | Intrinsics + gravity from one image, deep learning plus geometric optimization; accepts a known focal as a prior | Scene-only, knows nothing about the subject; PyTorch |
| [fSpy](https://github.com/stuffmatic/fSpy) | Desktop app: user drags vanishing lines, gets focal, orientation and position for Blender | Manual, needs straight edges, GPL, ignores people |
| [Fried et al. 2016](https://gfx.cs.princeton.edu/pubs/Fried_2016_PMO/fried2016-portraits.pdf) | Fits a 3D head and full-perspective camera to a portrait to estimate and edit camera distance | Research method, faces only |
| [CineScale camera-angle classifier](https://cinescale.github.io/camera_al) | Classifies film frames as Overhead / High / Neutral / Low / Dutch | Labels only, no numbers or 3D |
| AI "change camera angle" tools (VEED, Qwen-based editors) | Generate a new image from another viewpoint | Generation, not measurement |

**Where CamPose is different:** metric, explainable geometry (every number traces back to landmarks, focal length and a stated assumption); runs offline on a phone; shows its own error (reprojection overlay and ± ranges); MIT/Apache-friendly stack.

## Scope

The MVP answers "where was the camera?" for one main subject in one photo. The extended version adds the 3D rebuild and a what-if virtual camera.

**MVP outputs (per photo, main subject only)**

- Camera pitch (tilt up/down vs. horizon) and roll (Dutch tilt), in degrees
- Focal length used: 35mm-equivalent and pixels, horizontal/vertical FOV, and its source (EXIF, preset, or estimated)
- Distance from camera to the subject's eyes (3D and horizontal), in meters
- Camera height relative to the subject's eye level, plus height above the ground when the feet are visible
- Elevation angle (camera above/below the eye line) and azimuth around the subject (front, 3/4, profile, behind; subject's left or right)
- Labels: eye level / high / low / overhead / worm's-eye; Dutch angle when |roll| > 5°; framing scale (extreme close-up to long shot); a "selfie distortion" flag when distance < 0.6 m
- A ± range on every number, plus the assumptions behind it (subject height, upright posture, lens)
- 2D overlay on the photo: landmarks, the estimated horizon line, and the eye-level line

**Extended outputs**

- 3D scene: ground grid, skeleton, face mesh (wireframe or textured from the photo), camera frustum with the photo on its image plane
- Camera view: render from the solved camera at the same intrinsics, blended over the photo, to check alignment
- Virtual camera: drag the camera or switch lens (13 / 26 / 52 / 77 / 120 mm eq.) and see what that shot would look like, including a dolly-zoom mode that keeps the subject the same size
- Export: JSON result, plus camera parameters for Blender/three.js (copy button)

**Non-goals for v1:** multiple subjects (pick the largest; others optional later), video, people not standing on the same ground, heavy fisheye images, identity recognition. Nothing is uploaded to a server.

## Architecture

All logic lives in a Rust workspace compiled to `wasm32-unknown-unknown`. The browser only decodes the image, runs the Rust module in a Web Worker, and draws with raw WebGL2. There is no JS framework, no bundler and no npm runtime dependency.

&#91;embedded content: runtime architecture · 3 lanes, 9 parts\]

The models run once per photo. Changing the lens or subject height re-runs only the geometry (dashed arrow).

**Repository layout**

```
campose/
  Cargo.toml                 # workspace
  crates/
    core/    # zero-dependency: linalg, camera model, EXIF, PnP, solvers, metrics, JSON writer
    infer/   # model wrappers: preprocessing, tract-onnx run, decoding, ROI transforms
    wasm/    # wasm-bindgen API: init(models), analyze(...), scene buffers
    cli/     # native binary: campose analyze photo.jpg --json (tests, eval, batch)
  web/
    index.html  app.js  worker.js  ui.css
    viewer/  gl.js  shaders.js  scene.js  orbit.js
    models/  # .onnx files, lazy-fetched, cached in Cache Storage
  tools/     # Python, dev-only: convert_models.py, dump_golden.py, eval.py
  testdata/  # synthetic fixtures, golden landmark JSON, ground-truth photo set
```

**Dependency budget.** Every addition beyond this table needs a written reason in the PR.

| Dependency | Crate / place | Why | Shipped? |
| --- | --- | --- | --- |
| `tract-onnx` | infer | Pure-Rust ONNX inference that builds for wasm32 (no C++ runtime) | Yes. Optional later: replace with a custom executor (M9) |
| `wasm-bindgen` (+ `js-sys`) | wasm | JS/wasm glue | Yes. Could be dropped for raw `extern "C"` exports + manual memory |
| `zune-jpeg`, `png` | cli only, behind a feature | Decode images for native tests | No (native only) |
| Python: `onnx`, `tf2onnx`, `mediapipe`, `numpy` | tools/ | One-time model conversion and golden outputs | No (dev only) |
| `wasm-opt` (binaryen) | build step | Shrink the wasm | No (build only) |

**Explicitly not used:** OpenCV, `nalgebra` (write a \~600-line `linalg` module instead: Vec3, Mat3, Mat4, quaternion, 3x3 SVD/eigen via Jacobi, small dense Levenberg–Marquardt), `image` crate in wasm (the browser decodes), `serde` (hand-written JSON writer), Three.js, React, the MediaPipe JS runtime.

**Runtime flow:** main thread reads the file bytes → `createImageBitmap` decodes and applies EXIF orientation → RGBA pixels plus the raw bytes (for EXIF) go to the worker → Rust runs detect → landmarks → solve → returns result JSON and Float32Array scene buffers → main thread draws the overlay and the 3D view. Re-solving after a UI change (lens, subject height) skips inference and takes under 5 ms.

## Stage 1: Input and camera intrinsics

Focal length in pixels is the single most important input. It is also how phone vs. camera and wide vs. normal mode are handled: they are just different intrinsics. Read it from EXIF when possible, ask for a lens preset otherwise, and estimate it only as a last resort.

**EXIF parser (crates/core/exif.rs, no dependencies).** Parse JPEG APP1 `Exif\0\0` → TIFF header (`II`/`MM` byte order) → IFD0 → Exif sub-IFD (tag 0x8769). Read these tags:

| Tag | ID | Use |
| --- | --- | --- |
| Make, Model | 0x010F, 0x0110 | Phone vs. camera, lookup table |
| Orientation | 0x0112 | Swap W/H for values 5–8 |
| FocalLength | 0x920A | Physical mm (rational) |
| FocalLengthIn35mmFilm | 0xA405 | Preferred source |
| FocalPlaneX/YResolution + unit | 0xA20E, 0xA20F, 0xA210 | Exact sensor pixel pitch when present |
| PixelX/YDimension | 0xA002, 0xA003 | Detect resize vs. crop |
| LensModel | 0xA434 | Lens id, e.g. "back triple camera", "front camera" |
| DigitalZoomRatio | 0xA404 | Digital crop |

Support JPEG first. PNG `eXIf` and WebP `EXIF` chunks come next. HEIC is a stretch goal: only Safari decodes it, so other browsers should show "convert to JPEG".

**Focal length in pixels**, in priority order:

1. FocalPlaneXResolution present: f\_px = FocalLength\_mm × pixels-per-mm, scaled by (decoded width / PixelXDimension).
2. FocalLengthIn35mmFilm present. The 35mm equivalent is defined on the diagonal, so the long-side constant depends on the native sensor aspect. Assume a crop keeps the long side.

```latex
f_{px} = f_{35} \cdot \frac{W_{long}}{L}, \qquad L = 34.6\,\text{mm (4:3 native: phones)}, \quad L = 36\,\text{mm (3:2 native: most cameras)}
```

3. Only FocalLength\_mm plus Make/Model: use a hand-maintained `sensors.json` (model → sensor width in mm).
4. No EXIF (screenshots, social media): the user picks a preset. Default "Phone 1x", shown as an assumption with a wide ± range.
5. "Estimate": a 1-D search over f (12–200 mm eq.) minimizing body-PnP reprojection error. Trust it only when the error curve has a clear minimum (close full-body or wide shots); otherwise fall back to option 4 and say so.

**Lens presets (35mm-equivalent; typical values, user-editable):** Phone 0.5x ultrawide 13 mm · Phone 1x 26 mm · Phone 2x 52 mm · Phone 3x 77 mm · Phone 5x 120 mm · Front/selfie 24 mm · Camera 35 mm · Camera 50 mm · Camera 85 mm · Custom.

**Other intrinsics rules**

- Principal point = image center. Square pixels, no skew.
- Treat the image as pinhole: phones correct ultrawide distortion in-camera. Keep a `k1` field in the camera struct (default 0) for later.
- Crop detection: if the decoded aspect ratio differs from PixelX/YDimension, the image was cropped. Warn that the center and focal may be off, and widen the uncertainty.
- Mirror flag: LensModel containing "front" usually means a selfie. Many phones save these mirrored, so add a "photo is mirrored" toggle that swaps left/right in the azimuth output.

## Stage 2: Landmark models and inference

Use Google's MediaPipe models (BlazeFace + Face Mesh, BlazePose GHUM), converted once to ONNX and run with tract inside Rust. They are small, mobile-grade, Apache-2.0 per their model cards (confirm each card before shipping), and BlazePose already outputs metric 3D "world" landmarks, which the body solver needs.

| Model | Source | Output used | Notes |
| --- | --- | --- | --- |
| Face detector (BlazeFace short-range) | inside `face_landmarker.task` (a zip) | box + 6 keypoints | Seeds the face ROI and its rotation |
| Face landmarks | inside `face_landmarker.task` | 478 points (468 mesh + 10 iris), x/y in pixels, relative z | Skip the blendshape model |
| Pose detector | inside `pose_landmarker_{lite,full,heavy}.task` | box + alignment keypoints | Seeds the body ROI |
| Pose landmarks (BlazePose GHUM 3D) | same bundle | 33 image landmarks + visibility, 33 world landmarks in meters (hip-centered) | Default `full`; `lite` for low-end phones; ignore segmentation |
| Canonical face model | MediaPipe `canonical_face_model.obj` | 468 metric vertices, triangles, UVs | Rigid PnP model and mesh topology |

Fallback if conversion or op support fails: RTMPose-t + RTMDet-nano (Apache-2.0, official ONNX exports). They give 17 COCO keypoints in 2D only, so the body solver would need a mean 3D skeleton prior. Accuracy is worse, so keep this as plan B only.

**Conversion (tools/convert\_models.py, dev-only)**

1. Unzip each `.task`, list the `.tflite` files, print input/output shapes and dtypes. Do not hardcode sizes: read them.
2. `python -m tf2onnx.convert --tflite X.tflite --output X.onnx --opset 13`. Convert float16 weights to float32 if tract complains.
3. Run tract's CLI (`tract X.onnx dump`) and confirm every op is supported. Do this in week 1, because it is the biggest technical risk.
4. Golden parity: run MediaPipe Python on 30 test images and dump detections and landmarks to `testdata/golden/*.json`. The Rust pipeline must match within 1.0 px for 2D landmarks and 5 mm for world landmarks.
5. Write `web/models/manifest.json` with name, URL, size and SHA-256. The web app verifies the hash after download.

**Pre- and post-processing to replicate exactly.** Read MediaPipe's graph `.pbtxt` files (pose detection, pose landmark by ROI, face detection short range, face landmark) and copy their constants into Rust. Do not guess them.

- Detector input: letterbox to the model size, normalize to the range the model expects (\[-1, 1\] or \[0, 1\]; read it from the graph).
- SSD anchors generated from the graph's anchor options; decode boxes and keypoints; sigmoid scores; weighted NMS.
- ROI: center, size and rotation from the detector keypoints, then the graph's scale factors. Rotation makes the body or face upright in the crop.
- Crop: affine warp with bilinear sampling (own code, about 60 lines).
- Landmark decode: crop coordinates → inverse affine → image pixels. z uses the same scale as x. Apply the ROI rotation back to the world landmarks, as MediaPipe's world-landmark projection step does. Visibility = sigmoid(logit).
- Main subject = largest detection. Later: tap to choose another person.

**Inference runtime.** Build each tract model once at init (`into_optimized().into_runnable()`). Compile with `-C target-feature=+simd128`. Run single-threaded: wasm threads need COOP/COEP headers, which many hosts don't send. Fetch models lazily with a progress bar and keep them in Cache Storage.

**Performance targets (measure; these are goals, not known numbers):** face path under 400 ms and body path under 800 ms on a mid-range 2023 phone; default model download under 15 MB; wasm under 2 MB gzipped.

## Stage 3: Geometry solvers (crates/core)

The solver fits a rigid 3D model of the subject to the 2D landmarks with the known focal length (PnP). It then measures everything along a gravity "up" vector U, taken from posture or scene lines. Every number below follows from four things: K, the PnP pose, U, and the subject's scale.

**Conventions (put these at the top of `core/src/lib.rs` and test them)**

- Camera frame C (OpenCV): x right, y down, z forward. The camera sits at the origin.
- Subject frame W: origin on the ground under the mid-hip (full body) or at the eye midpoint (face only). Y = up (against gravity), Z = the subject's facing direction made horizontal, X = Y × Z (the subject's left). Units are meters; angles are radians internally and degrees in output.
- Rotations: rotation vector (Rodrigues) inside the optimizer, Mat3 elsewhere.

```latex
\mathbf{u} = \pi(\mathbf{K}(\mathbf{R}\mathbf{M} + \mathbf{t})), \qquad \mathbf{K} = \begin{bmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{bmatrix}, \qquad \pi([X,Y,Z]) = [X/Z,\; Y/Z]
```

**Face solver (close-ups)**

1. Rigid subset: choose 60–100 canonical-mesh vertices that barely move with expression (forehead, nose bridge and tip, inner/outer eye corners, cheekbones, temples). Exclude lips, eyelids and jaw. Pick them with data: run MediaPipe on 20 expressive faces, Procrustes-align them, and keep the 40% of vertices with the lowest variance. Commit the index list.
2. PnP with M = canonical vertices (convert cm to m): EPnP initialization, then Levenberg–Marquardt on 6 parameters minimizing the Huber-weighted reprojection error (δ = 2 px).
3. Scale check with the iris: when the iris is at least 8 px wide, depth from iris = f × 11.7 mm / iris diameter in px (MediaPipe's iris constant). Fuse it with the PnP depth by inverse variance and report both.
4. Face axes come from R. Verify the canonical model's axis directions (which way is up and out of the face) with a unit test on a frontal synthetic face. Don't assume.
5. Face-only gravity: assume head-up = world-up, with σ ≈ 10° pitch and 6° roll (people tilt their heads). If shoulders are visible, also run the pose model and prefer its roll.

**Body solver (half or full body)**

1. Model M = BlazePose world landmarks (meters, hip-centered). Scale s = H\_user / H\_est when the user enters the subject's height, where H\_est = eye-to-ankle length + 0.11 m. Otherwise s = 1, with a ±7% uncertainty.
2. Initial rotation: try both R0 = I and R0 = R\_ray (the rotation taking the optical axis onto the ray through the mid-hip pixel). Keep the one with lower reprojection error. It is not documented whether world landmarks are crop-relative, so let the data decide.
3. Translation given R is linear. Solve this 2N × 3 least-squares system in closed form (N = visible landmarks, with P = R M\_i, ũ = u − c\_x, ṽ = v − c\_y):

```latex
f\,t_x - \tilde{u}_i\,t_z = \tilde{u}_i P_{z,i} - f P_{x,i}, \qquad f\,t_y - \tilde{v}_i\,t_z = \tilde{v}_i P_{z,i} - f P_{y,i}
```

4. LM over (ω, t), weighted by visibility, Huber loss, plus a prior λ‖ω − ω₀‖² equal to σ ≈ 3°. Single-view depth is ambiguous, so the prior keeps the network's 3D orientation unless the 2D evidence clearly disagrees.

**Gravity (up vector U in camera coordinates)** is a weighted mean of the available candidates (weights 1/σ²):

| Candidate | When available | σ (starting values; tune on data) |
| --- | --- | --- |
| Scene vertical vanishing point (M8) | at least 3 vertical line segments agree | 1.5° |
| Standing line: mid-ankle → mid-shoulder | ankles visible (> 0.7), knees > 160° | 4° |
| Torso line: mid-hip → mid-shoulder | hips and shoulders visible | 8° |
| Face up axis | face found | 12° |

Add a weak roll prior toward 0 (σ = 10°), since most photos are near level. Report the spread between candidates as a confidence signal. Add a UI toggle "subject is not upright" that drops the posture candidates.

**Output metrics.** Let E = eye midpoint and F = ground point under the feet, both in camera coordinates, and ẑ = (0, 0, 1).

```latex
\theta_{pitch} = \arcsin(-\hat{z}\cdot U) \;(+ = \text{looking down}), \quad \phi_{roll} = \operatorname{atan2}(U_x, -U_y), \quad \Delta h_{eye} = -U\cdot E, \quad h_{ground} = -U\cdot F
```

```latex
d_h = \lVert E - (U\cdot E)\,U \rVert, \qquad \varepsilon = \operatorname{atan2}(\Delta h_{eye},\, d_h), \qquad \alpha = \operatorname{atan2}(\mathbf{c}\cdot X_W,\; \mathbf{c}\cdot Z_W),\; \mathbf{c} = -E_{\perp U}
```

- Facing direction Z\_W: body forward = normalize(left × U), where left = mean of (L shoulder − R shoulder) and (L hip − R hip). Face forward comes from the face R. Report head yaw relative to the body separately.
- Horizon line in the image: l = K⁻ᵀ U (homogeneous line). Draw it; it is the best visual check.
- Elevation labels (defaults, configurable): |ε| < 7° eye level · 7–30° high/low · 30–60° steep high/low · > 60° overhead / < −60° worm's-eye. Dutch angle when |roll| > 5°.
- Azimuth labels: 0° front · ±30–60° three-quarter · ±90° profile · ±180° behind. Positive = the subject's left (swap when the mirrored flag is set).
- Framing scale from the lowest visible landmark: eyes/mouth only = extreme close-up · shoulders = close-up · chest = medium close-up · waist = medium · knees = medium long · feet = full · subject < 50% of frame height = long.

**Uncertainty.** Monte Carlo with 64 samples, warm-started LM (5 iterations each, about 50 ms total). Perturb focal (σ = 2% EXIF, 10% preset, 25% estimated), subject scale (4% with height given, 7% without), landmarks (model-reported or 1% of the box) and U (per-source σ). Report the median and the 10–90% interval for every number.

## Stage 4: 3D reconstruction viewer

Rust builds the scene in the subject frame W and hands flat vertex buffers to a hand-written WebGL2 renderer of about 500 lines of JS. The viewer has three modes: orbit (see the setup), camera view (check the fit against the photo) and virtual camera (try other shots).

**Scene building (core/scene.rs).** For any point in camera coordinates, p\_W = R\_WC (p\_C − o\_C). The rows of R\_WC are X\_W, Y\_W and Z\_W written in C, and o\_C is W's origin in C. The camera's position in W is R\_WC(−o\_C). Emit:

- Skeleton: 33 joints (points) + BlazePose's bone connection list (lines). Bones are colored by side.
- Face mesh: canonical triangle topology. Vertices come either from the rigid PnP fit or from the detected landmarks back-projected at the solved depth (keeps the expression; this is the default). Texture: UV of vertex i = its pixel position (u/W, v/H) in the photo, which gives a photo-textured face for free.
- Ground grid at y = 0, only when feet were seen. Otherwise a faint grid labeled "assumed floor (eye height 1.60 m)".
- Camera: a frustum built from K, a photo-textured image plane at its far end, a dashed line from the lens to the eyes labeled with ε and the distance, and a translucent disc at the camera's height to show above/below eye level.

**Renderer (web/viewer/gl.js).** Three programs: screen-space thick lines (instanced quads), unlit textured triangles, and instanced point sprites. Depth test on, `antialias: true`, grid fading with distance. Labels are HTML elements positioned by projecting 3D anchor points each frame.

**Projection from intrinsics (OpenGL clip space, camera looking down −Z):**

```latex
\mathbf{P} = \begin{bmatrix} 2f/W & 0 & 1 - 2c_x/W & 0 \\ 0 & 2f/H & 2c_y/H - 1 & 0 \\ 0 & 0 & -\frac{n_f + n_n}{n_f - n_n} & -\frac{2 n_f n_n}{n_f - n_n} \\ 0 & 0 & -1 & 0 \end{bmatrix}, \qquad \mathbf{V}_{gl} = \operatorname{diag}(1,-1,-1,1)\,\begin{bmatrix} R_{WC}^{\top} & o_C \\ 0 & 1 \end{bmatrix}
```

Required unit test: projecting random points through V\_gl, P and the viewport must land within 0.01 px of the OpenCV projection u = π(K p\_C). Sign mistakes here are the most common bug.

**Views**

1. Orbit: free orbit/zoom/pan around the subject; frustum, skeleton, face mesh and labels visible. Click the camera to jump to camera view.
2. Camera view: render from the solved V and P at the photo's aspect ratio, over the photo, with an opacity slider. Show the mean reprojection error in px. This is the honesty check: if the skeleton doesn't line up, the numbers are wrong.
3. Virtual camera: a copy of the solved camera. Controls: orbit around the subject at a fixed distance, a height slider, and lens buttons (13 / 26 / 52 / 77 / 120 mm eq.). A dolly-zoom toggle keeps the subject the same size when the lens changes (d′ = d × f′/f). A picture-in-picture panel shows what this camera would capture. The metrics panel updates live by calling the same Rust metric functions with the virtual pose.

**Export:** copy buttons for the result JSON, a Blender camera (location, rotation, lens mm with sensor width 36, sensor fit) and a three.js PerspectiveCamera snippet.

## UI and UX

One page, no login, no upload: drop a photo, get numbers within about a second, then open the 3D view. Plain HTML/CSS/JS ES modules, mobile-first, dark and light themes.

1. **Landing:** a drop zone plus a file picker that accepts JPEG/PNG/WebP (and HEIC on Safari), the line "Runs on your device. Nothing is uploaded." and three sample photos (selfie, portrait, full body) to try.
2. **First load:** a progress bar for the model download and a note that later visits use the cache.
3. **Results layout:** the photo with the overlay on the left (desktop) or on top (phone); the results card beside it.
   - Overlay toggles: landmarks, horizon line, eye-level line, subject box.
   - Results card, headline first: "Camera 0.42 m above eye level, tilted 12° down, 2.3 m away, 3/4 from the subject's left. High angle, medium shot."
   - Then a table of each value with its ± range, and the assumption chips: Lens (from EXIF: iPhone 15 Pro 1x, 24 mm), Subject height (1.70 m assumed, editable), Upright (yes), Mirrored (no).
   - Editing any chip re-solves instantly without rerunning the models.
4. **Warnings, in plain words:** no EXIF (pick your lens) · photo looks cropped · subject is sitting or lying (posture-based angles are unreliable) · face too small for the iris check · strong disagreement between gravity cues.
5. **3D tab:** orbit / camera view / virtual camera tabs, lens buttons, a dolly-zoom toggle, an opacity slider, and copy buttons for export.
6. **Accessibility:** every number is also in text; keyboard orbit controls; color is never the only cue.

The tool works offline after the first load. A service worker is optional (not needed for the cache).

## Data contracts

The wasm module exposes four functions. Inference runs once per photo; `resolve` re-runs only the geometry when the user changes an assumption.

```rust
// crates/wasm/src/lib.rs
#[wasm_bindgen] pub fn init(models: ModelBytes) -> Result<(), JsError>;           // build tract plans once
#[wasm_bindgen] pub fn analyze(rgba: &[u8], w: u32, h: u32, file_bytes: &[u8],
                               opts_json: &str) -> Result<String, JsError>;      // detect + landmarks + solve
#[wasm_bindgen] pub fn resolve(opts_json: &str) -> Result<String, JsError>;       // geometry only, cached landmarks
#[wasm_bindgen] pub fn scene_buffers(view: &str) -> SceneBuffers;                 // Float32Array/Uint32Array views
```

Options: `{ "lens": {"mode": "auto|preset|custom|estimate", "f35": 26}, "subject_height_m": null, "upright": true, "mirrored": null, "model": "full", "mc_samples": 64 }`

Result (every measured value is `{ "v": median, "lo": p10, "hi": p90 }`; units in the key name):

```json
{
  "version": 1,
  "image": { "w": 3024, "h": 4032, "cropped": false, "mirrored": false },
  "intrinsics": { "f_px": 2960.1, "f35_mm": 24, "hfov_deg": 54.1, "vfov_deg": 68.6,
                  "cx": 1512, "cy": 2016, "source": "exif_35mm", "device": "Apple iPhone 15 Pro" },
  "mode": "body",
  "camera": {
    "pitch_deg": { "v": 12.3, "lo": 10.1, "hi": 14.6 },
    "roll_deg":  { "v": -1.2, "lo": -2.5, "hi": 0.1 },
    "height_above_eye_m": { "v": 0.42, "lo": 0.31, "hi": 0.55 },
    "height_above_ground_m": { "v": 2.02, "lo": 1.88, "hi": 2.17 },
    "distance_to_eyes_m": { "v": 2.35, "lo": 2.15, "hi": 2.58 },
    "horizontal_distance_m": { "v": 2.31, "lo": 2.11, "hi": 2.53 },
    "elevation_deg": { "v": 10.3, "lo": 7.9, "hi": 12.8 },
    "azimuth_deg": { "v": 38.0, "lo": 31.0, "hi": 45.0 }
  },
  "labels": { "angle": "high", "dutch": false, "side": "three_quarter_left", "framing": "medium", "selfie_distortion": false },
  "subject": { "height_m": 1.70, "height_source": "assumed", "head_yaw_vs_body_deg": -15.0 },
  "gravity": { "sources": ["standing", "torso", "face"], "spread_deg": 3.1 },
  "fit": { "reproj_rmse_px": 4.2, "landmarks_used": 29 },
  "pose": { "R_wc": [9 numbers], "cam_pos_w": [3 numbers] },
  "overlay": { "landmarks_px": [...], "horizon": [a, b, c], "eye_line": [a, b, c] },
  "warnings": ["no_exif", "cropped", "not_upright", "gravity_disagreement"]
}
```

The numbers above are placeholders that show the format, not expected results. The native CLI prints the same JSON, so the web app, tests and eval script share one contract.

## Validation and accuracy targets

Accuracy is proven in four layers. The home-made ground-truth photo set (layer 4) is the one that decides whether the tool is good.

1. **Synthetic geometry tests (core, no models, run in CI).** Random cameras: f 13–120 mm eq., pitch −60° to +80°, roll ±20°, distance 0.3–8 m, height 0.2–4 m. Subjects: canonical face, mean BlazePose skeleton. Project, add noise (0 / 1 / 3 px, 5% outliers), solve.
   - Zero noise must recover everything to 1e-6.
   - Convention checks: camera above the eyes → Δh > 0; looking down → pitch > 0; subject facing the camera → azimuth ≈ 0; subject turned to show their left side → azimuth > 0; GL projection = CV projection.
2. **Golden inference parity.** Rust landmarks vs. MediaPipe Python on 30 images (thresholds in Stage 2).
3. **Public datasets with ground-truth cameras (evaluation only; check each license, most are research/non-commercial):** SPEC-SYN and SPEC-MTP (pitch, roll, FOV of people photos), BEDLAM (synthetic people, ground-truth cameras, varied focal), 3DPW and EMDB (phone video with camera poses), BIWI head pose (faces).
4. **Home-made ground-truth set, about 60 photos, in `testdata/gt/`.** Tripod with a tape-measured lens height, a phone level app for pitch and roll, a subject of known height on a floor mark at a measured distance, and azimuth marks at 0/45/90°. Vary the lens (0.5x, 1x, 3x, a camera at 50 mm), height (0.3 / 1.0 / 1.6 / 2.2 m), distance (0.5–5 m), framing (close-up to full body) and setting (indoor/outdoor). Labels go in `labels.csv`.

`tools/eval.py` runs the CLI over a set and writes a markdown report: error per metric and bucket, interval coverage, and a gallery of the 10 worst cases with overlays.

**Initial targets (revise after the first eval run):**

| Case | Metric | Target |
| --- | --- | --- |
| Full body, EXIF focal, height given, ≤ 5 m | pitch / roll MAE | ≤ 3° / ≤ 2° |
| same | camera height above ground | ≤ 10 cm MAE |
| same | distance to eyes | ≤ 7% |
| same | azimuth | ≤ 10° |
| Face close-up 0.3–1.5 m, EXIF focal | distance | ≤ 8% |
| same | camera height vs. eye level | ≤ 5 cm |
| same | pitch (limited by head tilt) | ≤ 6° |
| Any case, preset lens | true value inside the 10–90% interval | ≥ 80% of photos |

## Milestones for the AI agent

Build in this order. Each milestone ends with its acceptance check passing in CI or in a written report. M2 is the go/no-go gate for the inference stack.

1. **M0 Scaffold.** Cargo workspace, `scripts/build-web.sh` (cargo build for wasm32 with simd128 → wasm-bindgen → wasm-opt), a static `web/` that loads the wasm, and CI running test + clippy + the wasm build.
   - Accept: `cargo test` is green, and the page prints a value computed in Rust.
2. **M1 Core math, EXIF, intrinsics.** linalg, camera model, EXIF parser (fixtures: EXIF blocks from iPhone, Pixel, Samsung, Canon and Sony files with the image data stripped), EPnP + LM, the full synthetic test suite.
   - Accept: all layer-1 tests pass, and EXIF fixtures give the correct f\_px.
3. **M2 Models + tract (gate).** Convert the models, check ops, run detectors and landmarks in the CLI, and pass the golden parity tests. Log native timings.
   - Accept: parity thresholds met. If tract can't run a model, stop and choose between candle, a custom executor or the RTMPose fallback before continuing. Write that decision in `docs/decisions.md`.
4. **M3 Face pipeline end to end (CLI).** Detection → 478 landmarks → rigid-subset PnP → iris fusion → metrics → JSON.
   - Accept: within 1.5× of the face targets on the GT close-ups.
5. **M4 Body pipeline (CLI).** BlazePose → body PnP → gravity fusion → all metrics → Monte Carlo ranges. Run face + body together on medium shots (face for distance, body for gravity).
   - Accept: within 1.5× of the body targets on the GT full-body photos.
6. **M5 Web MVP.** Worker, model fetch + cache with hash check, overlay, results card, assumption chips with instant re-solve, warnings.
   - Accept: works in Chrome, Firefox and Safari on desktop, iOS Safari and Android Chrome; works offline after the first load; timings measured and logged.
7. **M6 3D viewer.** Orbit, camera view with overlay and reprojection error, virtual camera with lens buttons and dolly zoom, export buttons.
   - Accept: the GL-vs-CV projection test passes; the camera view lines up visually on 10 sample photos.
8. **M7 Evaluation and tuning.** Full eval on the GT set and one public set, tune the σ values and thresholds, fix the worst cases, commit the report.
   - Accept: targets met, or each gap documented with its cause.
9. **M8 Scene gravity.** Line-segment detection (own gradient-based detector, no OpenCV) and vertical vanishing point by RANSAC with known f, fused into U.
   - Accept: pitch/roll improve on the indoor GT subset, with no regression outdoors.
10. **M9 Optional.** Custom inference executor replacing tract (only if the op list is small), multiple people, video, HEIC parsing.

## Risks, edge cases and rules for the agent

The biggest risks are the inference stack (M2) and photos without EXIF. Both have a planned fallback. The rest are edge cases the UI must flag rather than hide.

| Risk / edge case | Effect | Mitigation |
| --- | --- | --- |
| tract lacks an op in the converted models | No inference in Rust | Find out in week 1 (M2 gate); fall back to candle, a custom executor, or RTMPose |
| EXIF stripped (screenshots, social media, messaging apps) | Focal unknown → distance and height scale wrong | Lens preset prompt, estimate mode, wide ± ranges |
| Cropped image | Principal point off-center, focal mismatch | Crop detection warning, wider σ |
| Subject sitting, lying, leaning or on stairs | Posture-based gravity wrong | "Not upright" toggle, gravity-spread warning, scene vanishing points (M8) |
| Head tilted in a face-only shot | Pitch/roll error up to the tilt angle | Shown as a stated assumption; use shoulders when visible |
| Mirrored selfie | Left/right swapped | Front-camera detection + toggle |
| Child or very tall/short subject | Scale error → distance and height off | Subject-height input; children need a different prior (open question) |
| Strong lens distortion (action cams, fisheye) | PnP biased at the edges | Out of scope v1; `k1` field reserved |
| World-landmark axis convention undocumented | Wrong body orientation | Verify empirically in M2 with test images; never assume |

**Open questions for the owner**

- License target for the app: MIT/Apache only? This decides whether research-licensed datasets may be used even for evaluation.
- Is multi-person support needed in v1, or is the largest person enough?
- Hosting: any static host works. Is there a preferred one (GitHub Pages, Cloudflare Pages)?

**Rules for the AI agent**

1. Read this whole plan before writing code. Follow the milestone order; don't start the viewer before M4 passes.
2. Never guess model constants, axis conventions or anchor options: read them from MediaPipe's graph files or measure them, and write down where each came from.
3. Every geometric convention gets a synthetic unit test before real images are used.
4. Don't add a dependency outside the budget table without a one-paragraph justification in `docs/decisions.md`.
5. Every number shown to users comes with its source (EXIF, preset, assumed, measured) and a range. Never present a guess as a measurement.
6. Keep `crates/core` free of dependencies and of I/O, so it builds both natively and for wasm.
7. At the end of each milestone, write a short report: what works, measured numbers, known issues.

## Sources

- [SPEC: Seeing People in the Wild with an Estimated Camera (ICCV 2021)](https://spec.is.tue.mpg.de/) · [paper](https://arxiv.org/pdf/2110.00620)
- [CameraHMR: Aligning People with Perspective (2024)](https://arxiv.org/pdf/2411.08128)
- [GeoCalib (ECCV 2024)](https://github.com/cvg/GeoCalib)
- [fSpy](https://github.com/stuffmatic/fSpy)
- [Fried et al., Perspective-aware Manipulation of Portrait Photos (SIGGRAPH 2016)](https://gfx.cs.princeton.edu/pubs/Fried_2016_PMO/fried2016-portraits.pdf)
- [CineScale camera feature recognition](https://cinescale.github.io/camera_al)
- MediaPipe Tasks Vision package (`@mediapipe/tasks-vision` 1.2.3 README: model URLs and Face/Pose Landmarker APIs), inspected from npm
