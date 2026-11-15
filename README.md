# Lucidity

_Camera pose from a single photo (formerly CamPose Web)._

Find where the camera was from a photo of a person: tilt, roll, height relative to the
subject's eyes and the ground, distance, side angle and lens. Everything runs in the browser
and photos never leave your device.

1. **Batch analysis.** Add photos or a whole folder. Every photo gets a card with its
   camera-angle label, a side-view diagram and its key numbers. Filter by angle, sort by
   distance, height or tilt, and export everything as CSV or JSON. Every photo is analysed at
   the same size, so copies of one photo at different resolutions give the same result.
2. **Compare.** Pick two photos to see them side by side. You get a plain-words summary
   ("B was taken 0.3 m higher, 25° further round to the subject's left…"), side and top-down
   diagrams of both camera positions, and a table of differences.
3. **Detail.** Open a card to see the fitted skeleton, the horizon and the eye-level overlay,
   every value with its 10–90% range, and the assumptions behind it (lens, subject height,
   upright, mirrored). Change an assumption and the numbers update instantly, without
   re-running the models.
4. **3D scene.** Press **Build 3D scene**. The subject appears as a plain mannequin with the
   solved pose and head direction. The face can be a simple clay head, your face shape from
   the face mesh, or that shape coloured from the photo. A photo cut-out and skeleton are
   optional layers.
   - **Overview:** grab the photo's camera and drag it around the floor (Shift-drag moves it
     up or down). It re-frames like a photographer would, and a live inset shows what it now
     sees. A ghost marks the original position.
   - **Camera view:** look through the camera, game-style (drag to look, WASD/QE to move).
   - **Original shot:** the solved camera with the photo overlaid.

   Sliders also move the camera (side angle, distance, height, tilt, roll, lens, dolly zoom),
   and its numbers update live next to the original shot's.

The design plan is `CamPose Web — Camera Angle & Position from a Photo.md`, and the places
where the implementation differs from it are in [docs/decisions.md](docs/decisions.md).

## Run it

```bash
rustup target add wasm32-unknown-unknown
```

```bash
./scripts/build-web.sh
```

```bash
python scripts/serve.py
```

Then open http://localhost:8123/web/. Any static server works; the page must be served over
http(s), not opened as a file.

On the first visit the page downloads the landmark models (about 10 MB). Later visits load
them from the browser cache.

## Deploy (Vercel or any static host)

The server only hands out static files. All the heavy work runs in the visitor's browser:
decoding, landmark models, geometry (wasm) and 3D. `vercel.json` publishes the `web/` folder
as is, with no build step and no server functions.

`web/campose_core.wasm` is committed, so the host doesn't need Rust. After changing anything
in `crates/`, run `./scripts/build-web.sh` and commit the new wasm along with the code.

Other static hosts (Netlify, Cloudflare Pages, GitHub Pages): publish the `web/` directory.

## Tests

```bash
cargo test
```

The synthetic geometry tests (`crates/core/tests/synthetic.rs`) place a known camera around a
synthetic skeleton, project it, solve, and check that pitch, height, distance and azimuth are
recovered. They also check the sign conventions (looking down means pitch > 0, a camera on the
subject's left means azimuth > 0, and so on), the Monte Carlo ranges, iris depth and the focal
estimate.

## Layout

```
crates/core   zero-dependency Rust: linalg, EXIF, intrinsics, body PnP (LM), face depth,
              gravity fusion, metrics, Monte Carlo ranges, focal estimate, JSON
crates/wasm   raw extern "C" wasm exports
web/          index.html, app.js (queue, cards, detail), landmarks.js (MediaPipe),
              core.js (wasm wrapper), viewer/ (hand-written WebGL2 viewer)
testdata/     sample photos for local testing (MediaPipe test assets)
```

For a quick dev check, `?load=../testdata/samples/pose.jpg` on the page URL queues
same-origin images.

## Accuracy notes

- Distances scale with focal length. Photos with EXIF focal data (most phone photos straight
  from the camera) are best. Without EXIF, the lens is estimated from body perspective when
  the evidence is clear; otherwise a 26 mm phone lens is assumed and flagged.
- Tilt and roll come from posture (standing line, torso, face). A leaning or seated subject
  or a tilted head widens the ranges, and the app says so.
- Every number comes with a range and a source. A poor skeleton fit is flagged rather than
  hidden.
