# Decisions and deviations from the plan

Plan: `CamPose Web — Camera Angle & Position from a Photo.md`. This file records where the
implementation differs from it and why (plan rule 4).

## 1. Landmark inference runs on MediaPipe Tasks Vision (JS/WASM), not tract-onnx in Rust

**Plan:** convert the MediaPipe `.tflite` models to ONNX and run them with `tract` inside the
Rust wasm module (M2 gate). The MediaPipe JS runtime was listed as "explicitly not used".

**Done instead:** `web/landmarks.js` loads `@mediapipe/tasks-vision@1.2.3` from jsDelivr and
runs Pose Landmarker (lite by default, full optional) and Face Landmarker. The geometry stays
in Rust as planned (`crates/core`).

**Why:**
- The plan names model conversion and op support as the biggest technical risk (M2).
  The official runtime removes that risk, along with the need to re-implement anchors, ROI
  warps and decoding in Rust (Stage 2 pre/post-processing).
- Speed on weak laptops was an explicit requirement. Tasks Vision uses XNNPACK SIMD on the CPU
  and can use the GPU. Measured on the dev machine: about 60–90 ms per photo on CPU only, about
  30 ms with Auto, and about 1 s of one-time warm-up. Single-threaded tract would very likely be
  several times slower.
- It is the same family of models (BlazePose GHUM 3D, Face Mesh with iris), so the data
  the solver consumes (33 image + 33 world landmarks, 478 face points) is unchanged.

**Cost:** about 10 MB of runtime wasm from a CDN plus about 10 MB of models from Google
Storage, all cached by the browser after the first visit (models go into Cache Storage).
Nothing about the user's photos leaves the device.

**Revisit:** if a self-contained, offline-first build is required, vendor the npm package into
`web/vendor/` (Apache-2.0) instead of loading it from the CDN. The tract path (plan M2/M9)
can still replace it later; `detectPhoto()` is the only function that would change.

## 2. No wasm-bindgen; raw `extern "C"` exports

The plan allows this ("Could be dropped for raw extern "C" exports + manual memory"). The
interface is one flat `f64` input buffer (layout in `crates/core/src/solve.rs`, mirrored in
`web/core.js`) and a JSON string out. The wasm file is about 105 KB. The build needs only
`rustup target add wasm32-unknown-unknown`.

## 3. Crates

`crates/core` (zero dependencies, no I/O) and `crates/wasm` exist. `crates/infer` is not
needed (see 1). `crates/cli` is deferred. The synthetic test suite lives in
`crates/core/tests/synthetic.rs` and runs with `cargo test`.

## 4. Face solver: iris + eye spacing, not rigid canonical-mesh PnP

The canonical face model (`canonical_face_model.obj`) is not vendored. Face depth is fused by
inverse variance from:
- the iris diameter (11.7 mm, MediaPipe's constant; used when the iris is at least 8 px), and
- the 3D interpupillary distance from the Face Mesh points (63 mm ± 6%).

Face axes come from the landmark geometry (eye corners, forehead to chin), rotated onto the
camera ray through the eyes. When the body is also visible, the face depth and the body depth
are fused, and the whole posed body is scaled. Scaling about the camera keeps the reprojection
exact.

## 5. Focal estimate (plan Stage 1, option 5) is on by default without EXIF

When a photo has no focal length in EXIF and the user has not picked a lens, the solver runs a
1-D search over focal length (about 12–400 mm eq.) and keeps the focal length with the lowest
body reprojection error. It accepts the result only when the error curve has a clear minimum:
- the preset fits more than 35% worse than the best focal, or
- the preset's extra error (in quadrature, above the structural floor) is larger than the
  landmark noise.

The reported focal interval covers every focal length that fits within half the landmark noise
of the best fit, and that interval feeds the Monte Carlo focal σ. The UI shows the lens as
"≈ N mm, estimated from body perspective" together with its range.

## 6. Subject height model

The plan used eye-to-ankle length + 0.11 m. That length collapses for lunges and seated people,
so the code uses the segment chain instead: eyes→shoulder line, shoulders→hips, thigh, shin
(legs averaged), plus 0.11 m. Without a user-entered height, the body is scaled to the plan's
assumed 1.70 m (σ 7%). When the legs are out of frame, their lengths are only the network's
guesses, so body-scale σ rises to 15% (12% with a given height) and the iris dominates the
distance.

## 7. Reported value `v` is the nominal solve, not the Monte Carlo median

This keeps every number consistent with the 3D scene and the overlays. `lo` and `hi` are the
10th and 90th percentiles of the Monte Carlo samples, widened to include `v`.

## 8. App flow (owner request)

The flow is batch first: drop many photos or a whole folder, and each photo gets a card. Cards
can be filtered by angle and sorted by distance, height or tilt, and everything exports as
CSV or JSON. The 3D scene is built only when the user presses **Build 3D scene** on one
photo. That step runs one extra segmentation pass for the photo cut-out.

The viewer has three modes:
- **Fly:** game-style, drag to look, WASD/QE to move.
- **Orbit.**
- **Photo camera:** the exact solved camera with the photo overlaid at adjustable opacity.

It also has lens buttons, a dolly-zoom option and a live HUD with the virtual camera's
metrics. It redraws only on input, caps the device pixel ratio at 1.5 and asks for a
low-power GL context, all to stay light on weak laptops.

## Not done yet (plan milestones)

- M2 golden parity: not applicable while MediaPipe's own runtime is used.
- M7 evaluation on a ground-truth photo set, and tuning of the σ values. The rotation prior is
  exposed (`setRotPrior` in `web/core.js`) for that.
- M8 scene gravity from vertical vanishing points. Posture-only tilt can be off by about 10°
  when the subject leans; the wide ranges reflect this.
- PNG/WebP EXIF parsing is implemented but not tested on real device files. HEIC decodes in
  Safari only.
