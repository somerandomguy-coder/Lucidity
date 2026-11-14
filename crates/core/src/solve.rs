//! Stage 3: geometry solvers. Body PnP, face (iris / IPD) depth, gravity fusion,
//! output metrics, Monte Carlo uncertainty and scene buffers.
//!
//! Conventions (see lib.rs): camera frame C is OpenCV (x right, y down, z forward),
//! camera at the origin. U is the "up" unit vector (against gravity) written in C.

use crate::json::{fmt_num, Json};
use crate::linalg::*;

// ---- BlazePose landmark indices -------------------------------------------------
pub const NOSE: usize = 0;
pub const L_EYE: usize = 2;
pub const R_EYE: usize = 5;
pub const L_SH: usize = 11;
pub const R_SH: usize = 12;
pub const L_HIP: usize = 23;
pub const R_HIP: usize = 24;
pub const L_KNEE: usize = 25;
pub const R_KNEE: usize = 26;
pub const L_ANK: usize = 27;
pub const R_ANK: usize = 28;
pub const FEET: [usize; 6] = [27, 28, 29, 30, 31, 32];
pub const N_POSE: usize = 33;
pub const N_FACE: usize = 478;

// ---- Face Mesh landmark indices ---------------------------------------------------
const F_R_IRIS: usize = 468; // ring 469..472
const F_L_IRIS: usize = 473; // ring 474..477
const F_R_EYE_OUT: usize = 33;
const F_L_EYE_OUT: usize = 263;
const F_FOREHEAD: usize = 10;
const F_CHIN: usize = 152;

/// MediaPipe iris diameter constant (meters).
pub const IRIS_DIAM_M: f64 = 0.0117;
/// Mean adult interpupillary distance (meters).
pub const IPD_M: f64 = 0.063;
/// Eye height below the top of the head (meters), used for subject height.
pub const EYE_TO_TOP_M: f64 = 0.11;
pub const DEFAULT_SUBJECT_H: f64 = 1.70;

// ---- Input layout (flat f64 buffer shared with JS) --------------------------------
pub const HDR: usize = 16;
pub const OFF_POSE2D: usize = HDR; // 33 × (u, v, vis)
pub const OFF_WORLD: usize = OFF_POSE2D + N_POSE * 3; // 33 × (x, y, z) meters
pub const OFF_FACE: usize = OFF_WORLD + N_POSE * 3; // 478 × (u, v, z) pixels
pub const INPUT_LEN: usize = OFF_FACE + N_FACE * 3;

#[derive(Clone, Copy, Debug)]
pub struct Cam {
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
}
impl Cam {
    pub fn project(&self, p: V3) -> [f64; 2] {
        [self.f * p[0] / p[2] + self.cx, self.f * p[1] / p[2] + self.cy]
    }
    pub fn ray(&self, u: f64, v: f64) -> V3 {
        [(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0]
    }
}

#[derive(Clone, Debug)]
pub struct Pose {
    pub uv: [[f64; 2]; N_POSE],
    pub vis: [f64; N_POSE],
    pub world: [V3; N_POSE],
}

#[derive(Clone, Debug)]
pub struct Input {
    pub w: f64,
    pub h: f64,
    pub f: f64,
    pub f_sigma: f64,
    pub subject_h: Option<f64>,
    pub upright: bool,
    pub mirrored: bool,
    pub mc: usize,
    pub pose: Option<Pose>,
    pub face: Option<Vec<V3>>,
    pub no_exif: bool,
    pub cropped: bool,
    pub seed: u64,
    /// Allow estimating the focal length from the body fit (no EXIF, lens not chosen by the user).
    pub allow_estimate: bool,
}

pub fn parse_input(b: &[f64]) -> Option<Input> {
    if b.len() < INPUT_LEN {
        return None;
    }
    let pose = if b[8] > 0.5 {
        let mut p = Pose { uv: [[0.0; 2]; N_POSE], vis: [0.0; N_POSE], world: [[0.0; 3]; N_POSE] };
        for i in 0..N_POSE {
            p.uv[i] = [b[OFF_POSE2D + i * 3], b[OFF_POSE2D + i * 3 + 1]];
            p.vis[i] = b[OFF_POSE2D + i * 3 + 2];
            p.world[i] = [b[OFF_WORLD + i * 3], b[OFF_WORLD + i * 3 + 1], b[OFF_WORLD + i * 3 + 2]];
        }
        Some(p)
    } else {
        None
    };
    let face = if b[9] > 0.5 {
        Some((0..N_FACE).map(|i| [b[OFF_FACE + i * 3], b[OFF_FACE + i * 3 + 1], b[OFF_FACE + i * 3 + 2]]).collect())
    } else {
        None
    };
    Some(Input {
        w: b[0],
        h: b[1],
        f: b[2],
        f_sigma: b[3].max(0.0),
        subject_h: if b[4] > 0.5 { Some(b[4]) } else { None },
        upright: b[5] > 0.5,
        mirrored: b[6] > 0.5,
        mc: (b[7].max(0.0) as usize).min(512),
        pose,
        face,
        no_exif: b[10] > 0.5,
        cropped: b[11] > 0.5,
        seed: b[12] as u64,
        allow_estimate: b[13] > 0.5,
    })
}

// ---- RNG ------------------------------------------------------------------------
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---- Body PnP -------------------------------------------------------------------

/// Rotation prior σ (degrees, stored as f64 bits): keeps the pose network's 3D orientation
/// unless the 2D evidence clearly disagrees. Tunable at runtime for evaluation (plan M7).
static ROT_PRIOR_DEG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x4008000000000000); // 3.0
pub fn set_rot_prior_deg(d: f64) {
    ROT_PRIOR_DEG.store(d.max(0.01).to_bits(), std::sync::atomic::Ordering::Relaxed);
}
fn rot_prior() -> f64 {
    f64::from_bits(ROT_PRIOR_DEG.load(std::sync::atomic::Ordering::Relaxed)).to_radians()
}

/// Closed-form translation for a fixed rotation (weighted 2N×3 least squares).
pub fn solve_t(r: &M3, model: &[V3], uv: &[[f64; 2]], w: &[f64], cam: &Cam) -> Option<V3> {
    let mut a = [[0.0; 3]; 3];
    let mut b = [0.0; 3];
    let f = cam.f;
    for i in 0..model.len() {
        if w[i] <= 0.0 {
            continue;
        }
        let p = mat_vec(r, model[i]);
        let ut = uv[i][0] - cam.cx;
        let vt = uv[i][1] - cam.cy;
        let rows = [([f, 0.0, -ut], ut * p[2] - f * p[0]), ([0.0, f, -vt], vt * p[2] - f * p[1])];
        for (row, rhs) in rows.iter() {
            for j in 0..3 {
                for k in 0..3 {
                    a[j][k] += w[i] * row[j] * row[k];
                }
                b[j] += w[i] * row[j] * rhs;
            }
        }
    }
    solve_dense(a, b)
}

#[derive(Clone, Debug)]
pub struct Fit {
    pub r: M3,
    pub t: V3,
    pub rmse_px: f64,
    pub n: usize,
}

fn reproj_rmse(r: &M3, t: V3, model: &[V3], uv: &[[f64; 2]], w: &[f64], cam: &Cam) -> f64 {
    let mut s = 0.0;
    let mut n = 0.0;
    for i in 0..model.len() {
        if w[i] <= 0.0 {
            continue;
        }
        let p = add(mat_vec(r, model[i]), t);
        if p[2] <= 1e-6 {
            return f64::INFINITY;
        }
        let q = cam.project(p);
        s += w[i] * ((q[0] - uv[i][0]).powi(2) + (q[1] - uv[i][1]).powi(2));
        n += w[i];
    }
    if n <= 0.0 { f64::INFINITY } else { (s / n).sqrt() }
}

/// Levenberg–Marquardt over (ω, t) with R = exp(ω)·R0, Huber loss and a prior ‖ω‖/σ_ω.
pub fn lm_refine(
    model: &[V3],
    uv: &[[f64; 2]],
    w: &[f64],
    cam: &Cam,
    r0: M3,
    t0: V3,
    sigma_px: f64,
    sigma_rot: f64,
    iters: usize,
) -> (M3, V3) {
    let n = model.len();
    let delta = 2.0; // Huber threshold in units of sigma
    let mut p = [0.0, 0.0, 0.0, t0[0], t0[1], t0[2]];
    let mut hub = vec![1.0; n];
    let residuals = |p: &[f64; 6], hub: &[f64], out: &mut Vec<f64>| {
        out.clear();
        let r = mat_mul(&rodrigues([p[0], p[1], p[2]]), &r0);
        let t = [p[3], p[4], p[5]];
        for i in 0..n {
            if w[i] <= 0.0 {
                continue;
            }
            let q = add(mat_vec(&r, model[i]), t);
            let s = (w[i] * hub[i]).sqrt() / sigma_px;
            if q[2] <= 1e-4 {
                out.push(1e3 * s);
                out.push(1e3 * s);
                continue;
            }
            let pr = cam.project(q);
            out.push(s * (pr[0] - uv[i][0]));
            out.push(s * (pr[1] - uv[i][1]));
        }
        if sigma_rot > 0.0 {
            for k in 0..3 {
                out.push(p[k] / sigma_rot);
            }
        }
    };
    let mut lambda = 1e-3;
    let mut r = Vec::with_capacity(2 * n + 3);
    let mut rp = Vec::with_capacity(2 * n + 3);
    let mut rm = Vec::with_capacity(2 * n + 3);
    for _ in 0..iters {
        // IRLS Huber weights from current residuals
        {
            let rr = mat_mul(&rodrigues([p[0], p[1], p[2]]), &r0);
            let t = [p[3], p[4], p[5]];
            for i in 0..n {
                if w[i] <= 0.0 {
                    continue;
                }
                let q = add(mat_vec(&rr, model[i]), t);
                if q[2] <= 1e-4 {
                    hub[i] = 1.0;
                    continue;
                }
                let pr = cam.project(q);
                let e = ((pr[0] - uv[i][0]).powi(2) + (pr[1] - uv[i][1]).powi(2)).sqrt() / sigma_px;
                hub[i] = if e <= delta { 1.0 } else { delta / e };
            }
        }
        residuals(&p, &hub, &mut r);
        let cost: f64 = r.iter().map(|x| x * x).sum();
        let m = r.len();
        // numeric Jacobian (central differences)
        let mut jac = vec![[0.0f64; 6]; m];
        for k in 0..6 {
            let h = if k < 3 { 1e-6 } else { 1e-6 * (1.0 + p[5].abs()) };
            let mut pp = p;
            pp[k] += h;
            let mut pm = p;
            pm[k] -= h;
            residuals(&pp, &hub, &mut rp);
            residuals(&pm, &hub, &mut rm);
            for i in 0..m {
                jac[i][k] = (rp[i] - rm[i]) / (2.0 * h);
            }
        }
        let mut jtj = [[0.0; 6]; 6];
        let mut jtr = [0.0; 6];
        for i in 0..m {
            for a in 0..6 {
                jtr[a] += jac[i][a] * r[i];
                for b in a..6 {
                    jtj[a][b] += jac[i][a] * jac[i][b];
                }
            }
        }
        for a in 0..6 {
            for b in 0..a {
                jtj[a][b] = jtj[b][a];
            }
        }
        let mut improved = false;
        let mut step_norm = 0.0;
        for _ in 0..8 {
            let mut a = jtj;
            for d in 0..6 {
                a[d][d] += lambda * (jtj[d][d] + 1e-9);
            }
            let neg: [f64; 6] = [-jtr[0], -jtr[1], -jtr[2], -jtr[3], -jtr[4], -jtr[5]];
            if let Some(dx) = solve_dense(a, neg) {
                let mut p2 = p;
                for k in 0..6 {
                    p2[k] += dx[k];
                }
                residuals(&p2, &hub, &mut rp);
                let c2: f64 = rp.iter().map(|x| x * x).sum();
                if c2 < cost {
                    step_norm = dx.iter().map(|x| x * x).sum::<f64>().sqrt();
                    p = p2;
                    lambda = (lambda * 0.3).max(1e-9);
                    improved = true;
                    break;
                }
            }
            lambda *= 10.0;
        }
        if !improved || step_norm < 1e-10 {
            break;
        }
    }
    (mat_mul(&rodrigues([p[0], p[1], p[2]]), &r0), [p[3], p[4], p[5]])
}

pub fn pose_weights(pose: &Pose, w: f64, h: f64) -> Vec<f64> {
    let margin = 0.02 * w.max(h);
    (0..N_POSE)
        .map(|i| {
            let [u, v] = pose.uv[i];
            let inside = u >= -margin && v >= -margin && u <= w + margin && v <= h + margin;
            if inside && pose.vis[i] > 0.3 { pose.vis[i] } else { 0.0 }
        })
        .collect()
}

/// Standing-height estimate from world landmarks, independent of posture: sum of the
/// segment chain eyes → shoulders → hips → knees → ankles (legs averaged) + eye-to-top.
pub fn world_height(world: &[V3; N_POSE]) -> f64 {
    let eye = mid(world[L_EYE], world[R_EYE]);
    let sh = mid(world[L_SH], world[R_SH]);
    let hip = mid(world[L_HIP], world[R_HIP]);
    let thigh = 0.5 * (norm(sub(world[L_HIP], world[L_KNEE])) + norm(sub(world[R_HIP], world[R_KNEE])));
    let shin = 0.5 * (norm(sub(world[L_KNEE], world[L_ANK])) + norm(sub(world[R_KNEE], world[R_ANK])));
    // eye→shoulder-line runs a little forward of the spine; project it on the torso axis
    let torso = sub(sh, hip);
    let neck = dot(sub(eye, sh), normalize(torso)).max(0.0);
    neck + norm(torso) + thigh + shin + EYE_TO_TOP_M
}

fn landmark_sigma_px(pose: &Pose, w: &[f64]) -> f64 {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for i in 0..N_POSE {
        if w[i] > 0.0 {
            x0 = x0.min(pose.uv[i][0]);
            y0 = y0.min(pose.uv[i][1]);
            x1 = x1.max(pose.uv[i][0]);
            y1 = y1.max(pose.uv[i][1]);
        }
    }
    let diag = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
    (0.01 * diag).max(1.5)
}

/// Full body fit: try R0 = I and R0 = R_ray, keep the better one.
pub fn body_fit(model: &[V3], uv: &[[f64; 2]], w: &[f64], cam: &Cam, sigma_px: f64) -> Option<Fit> {
    let n = w.iter().filter(|&&x| x > 0.0).count();
    if n < 6 {
        return None;
    }
    let hip = mid2(uv[L_HIP], uv[R_HIP]);
    let r_ray = rot_between([0.0, 0.0, 1.0], cam.ray(hip[0], hip[1]));
    let sigma_rot = rot_prior();
    let mut best: Option<Fit> = None;
    for r0 in [I3, r_ray] {
        let t0 = match solve_t(&r0, model, uv, w, cam) {
            Some(t) if t[2] > 0.05 => t,
            _ => continue,
        };
        let (r, t) = lm_refine(model, uv, w, cam, r0, t0, sigma_px, sigma_rot, 30);
        let e = reproj_rmse(&r, t, model, uv, w, cam);
        if best.as_ref().map_or(true, |b| e < b.rmse_px) {
            best = Some(Fit { r, t, rmse_px: e, n });
        }
    }
    best
}

// ---- Face -----------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FaceEst {
    /// Depth (z) of the eye midpoint, meters.
    pub z: f64,
    pub z_sigma_rel: f64,
    pub eye_px: [f64; 2],
    pub iris_px: f64,
    pub used_iris: bool,
    /// Face axes in C as columns: X = subject's left, Y = up, Z = forward.
    pub axes: M3,
    pub bbox_h: f64,
    pub z_eyes_px: f64,
}

pub fn face_estimate(face: &[V3], cam: &Cam) -> Option<FaceEst> {
    if face.len() < N_FACE {
        return None;
    }
    let d2 = |a: usize, b: usize| ((face[a][0] - face[b][0]).powi(2) + (face[a][1] - face[b][1]).powi(2)).sqrt();
    let iris_r = d2(469, 471).max(d2(470, 472));
    let iris_l = d2(474, 476).max(d2(475, 477));
    let iris = iris_r.max(iris_l);
    let ipd3 = norm(sub(face[F_R_IRIS], face[F_L_IRIS]));
    if ipd3 < 2.0 {
        return None;
    }
    let z_ipd = cam.f * IPD_M / ipd3;
    let s_ipd = 0.06;
    let (z, s, used_iris) = if iris >= 8.0 {
        let z_ir = cam.f * IRIS_DIAM_M / iris;
        let s_ir = (0.045f64.powi(2) + (0.7 / iris).powi(2)).sqrt();
        let wi = 1.0 / (s_ir * s_ir);
        let wp = 1.0 / (s_ipd * s_ipd);
        ((z_ir * wi + z_ipd * wp) / (wi + wp), (1.0 / (wi + wp)).sqrt(), true)
    } else {
        (z_ipd, s_ipd, false)
    };
    let eye = mid(face[F_R_IRIS], face[F_L_IRIS]);
    let left = sub(face[F_L_EYE_OUT], face[F_R_EYE_OUT]);
    let up = sub(face[F_FOREHEAD], face[F_CHIN]);
    let y = normalize(up);
    let x = normalize(reject(left, y));
    let zf = cross(x, y);
    let weak = from_cols(x, y, zf);
    let r_ray = rot_between([0.0, 0.0, 1.0], cam.ray(eye[0], eye[1]));
    let axes = mat_mul(&r_ray, &weak);
    let (mut y0, mut y1) = (f64::MAX, f64::MIN);
    for p in face.iter().take(468) {
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    Some(FaceEst {
        z,
        z_sigma_rel: s,
        eye_px: [eye[0], eye[1]],
        iris_px: iris,
        used_iris,
        axes,
        bbox_h: y1 - y0,
        z_eyes_px: eye[2],
    })
}

fn col(m: &M3, j: usize) -> V3 {
    [m[0][j], m[1][j], m[2][j]]
}

// ---- Gravity --------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Gravity {
    pub u: V3,
    pub sigma: f64,
    pub sources: Vec<&'static str>,
    pub spread: f64,
}

pub fn pitch_roll(u: V3) -> (f64, f64) {
    ((-u[2]).clamp(-1.0, 1.0).asin(), u[0].atan2(-u[1]))
}
pub fn u_from_pitch_roll(pitch: f64, roll: f64) -> V3 {
    // inverse of pitch_roll: u = (sin r cos p, -cos r cos p, -sin p)
    [roll.sin() * pitch.cos(), -roll.cos() * pitch.cos(), -pitch.sin()]
}

pub fn fuse_gravity(cands: &[(V3, f64, &'static str)], roll_prior_sigma: f64) -> Gravity {
    if cands.is_empty() {
        return Gravity { u: [0.0, -1.0, 0.0], sigma: 15f64.to_radians(), sources: vec!["assumed_level"], spread: 0.0 };
    }
    let mut acc = [0.0; 3];
    let mut wsum = 0.0;
    for (u, s, _) in cands {
        let w = 1.0 / (s * s);
        acc = add(acc, scale(normalize(*u), w));
        wsum += w;
    }
    let mut u = normalize(acc);
    let sigma = (1.0 / wsum).sqrt();
    // weak roll prior toward 0
    if roll_prior_sigma > 0.0 {
        let (p, r) = pitch_roll(u);
        let k = roll_prior_sigma.powi(2) / (roll_prior_sigma.powi(2) + sigma * sigma);
        u = u_from_pitch_roll(p, r * k);
    }
    let mut spread: f64 = 0.0;
    for i in 0..cands.len() {
        for j in i + 1..cands.len() {
            spread = spread.max(angle(cands[i].0, cands[j].0));
        }
    }
    Gravity { u, sigma, sources: cands.iter().map(|c| c.2).collect(), spread }
}

fn joint_angle(a: V3, b: V3, c: V3) -> f64 {
    angle(sub(a, b), sub(c, b))
}

// ---- Metrics --------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    pub pitch: f64,
    pub roll: f64,
    pub h_eye: f64,
    pub h_ground: f64,
    pub dist: f64,
    pub d_h: f64,
    pub elev: f64,
    pub azim: f64,
}

/// E: eye midpoint in C; ground_h: U·F of the ground (None if unknown); facing: subject forward in C.
pub fn metrics(e: V3, ground_h: Option<f64>, u: V3, facing: V3, mirrored: bool) -> Metrics {
    let (pitch, roll) = pitch_roll(u);
    let h_eye = -dot(u, e);
    let d_h = norm(reject(e, u));
    let elev = h_eye.atan2(d_h);
    let zw = normalize(reject(facing, u));
    let xw = cross(u, zw);
    let c = scale(reject(e, u), -1.0);
    let mut azim = dot(c, xw).atan2(dot(c, zw));
    if mirrored {
        azim = -azim;
    }
    Metrics {
        pitch,
        roll,
        h_eye,
        h_ground: ground_h.map_or(f64::NAN, |g| -g),
        dist: norm(e),
        d_h,
        elev,
        azim,
    }
}

pub fn angle_label(elev_deg: f64) -> &'static str {
    let a = elev_deg.abs();
    if a < 7.0 {
        "eye_level"
    } else if elev_deg > 0.0 {
        if a < 30.0 { "high" } else if a < 60.0 { "steep_high" } else { "overhead" }
    } else if a < 30.0 {
        "low"
    } else if a < 60.0 {
        "steep_low"
    } else {
        "worms_eye"
    }
}

pub fn side_label(az_deg: f64) -> &'static str {
    let a = az_deg.abs();
    let left = az_deg > 0.0;
    if a < 20.0 {
        "front"
    } else if a < 67.5 {
        if left { "three_quarter_left" } else { "three_quarter_right" }
    } else if a < 135.0 {
        if left { "profile_left" } else { "profile_right" }
    } else {
        "behind"
    }
}

fn wrap_pi(a: f64) -> f64 {
    let mut a = a;
    while a > std::f64::consts::PI {
        a -= 2.0 * std::f64::consts::PI;
    }
    while a < -std::f64::consts::PI {
        a += 2.0 * std::f64::consts::PI;
    }
    a
}

// ---- Full analysis --------------------------------------------------------------

/// One deterministic solve (nominal or one Monte Carlo sample).
struct Solved {
    m: Metrics,
    e: V3,
    ground_h: f64,
    ground_seen: bool,
    u: V3,
    facing: V3,
    face_fwd: Option<V3>,
    body_fwd: Option<V3>,
    joints: Option<Vec<V3>>,
    fit: Option<Fit>,
    face_z: Option<f64>,
    scale: f64,
}

struct Perturb {
    f_mul: f64,
    s_mul: f64,
    face_mul: f64,
    uv_noise: Option<Vec<[f64; 2]>>,
    grav_rot: V3,
}

struct Ctx<'a> {
    inp: &'a Input,
    weights: Vec<f64>,
    sigma_px: f64,
    base_scale: f64,
    scale_sigma: f64,
    face: Option<FaceEst>,
    face_ok: bool,
    standing: bool,
    /// shoulders and hips confidently inside the frame (strict: no edge margin)
    torso_ok: bool,
    sitting_detected: bool,
}

fn solve_once(ctx: &Ctx, pt: &Perturb, warm: Option<&Fit>) -> Option<Solved> {
    let inp = ctx.inp;
    let cam = Cam { f: inp.f * pt.f_mul, cx: inp.w / 2.0, cy: inp.h / 2.0 };
    let mut body: Option<(Fit, Vec<V3>, f64)> = None;
    if let Some(pose) = &inp.pose {
        let s = ctx.base_scale * pt.s_mul;
        let model: Vec<V3> = pose.world.iter().map(|p| scale(*p, s)).collect();
        let uv: Vec<[f64; 2]> = match &pt.uv_noise {
            Some(nz) => (0..N_POSE).map(|i| [pose.uv[i][0] + nz[i][0], pose.uv[i][1] + nz[i][1]]).collect(),
            None => pose.uv.to_vec(),
        };
        let fit = match warm {
            Some(wf) => {
                let t0 = solve_t(&wf.r, &model, &uv, &ctx.weights, &cam).unwrap_or(wf.t);
                let (r, t) = lm_refine(&model, &uv, &ctx.weights, &cam, wf.r, t0, ctx.sigma_px, rot_prior(), 5);
                let e = reproj_rmse(&r, t, &model, &uv, &ctx.weights, &cam);
                Some(Fit { r, t, rmse_px: e, n: wf.n })
            }
            None => body_fit(&model, &uv, &ctx.weights, &cam, ctx.sigma_px),
        };
        if let Some(fit) = fit {
            let posed: Vec<V3> = model.iter().map(|m| add(mat_vec(&fit.r, *m), fit.t)).collect();
            body = Some((fit, posed, s));
        }
    }

    // Face depth (scales with focal length)
    let face_z = if ctx.face_ok {
        ctx.face.as_ref().map(|fe| fe.z * pt.f_mul * pt.face_mul)
    } else {
        None
    };

    // Combine depths and pick the eye point
    let (e, joints, scale_used, fit) = match (body, face_z, ctx.face.as_ref()) {
        (Some((fit, mut posed, s)), fz, fe) => {
            let e_body = mid(posed[L_EYE], posed[R_EYE]);
            let mut k = 1.0;
            if let (Some(fz), Some(fe)) = (fz, fe) {
                let ez = e_body[2];
                let wb = 1.0 / ctx.scale_sigma.powi(2);
                let wf = 1.0 / fe.z_sigma_rel.powi(2);
                let z = (ez * wb + fz * wf) / (wb + wf);
                k = z / ez;
            }
            for p in posed.iter_mut() {
                *p = scale(*p, k);
            }
            let e = mid(posed[L_EYE], posed[R_EYE]);
            let fit = Fit { r: fit.r, t: scale(fit.t, k), rmse_px: fit.rmse_px, n: fit.n };
            (e, Some(posed), s * k, Some(fit))
        }
        (None, Some(fz), Some(fe)) => {
            let e = scale(cam.ray(fe.eye_px[0], fe.eye_px[1]), fz);
            (e, None, 1.0, None)
        }
        _ => return None,
    };

    // Gravity candidates
    let mut cands: Vec<(V3, f64, &'static str)> = vec![];
    let mut body_fwd = None;
    if let Some(j) = &joints {
        let sh = mid(j[L_SH], j[R_SH]);
        let hip = mid(j[L_HIP], j[R_HIP]);
        if inp.upright {
            if ctx.standing {
                let ank = mid(j[L_ANK], j[R_ANK]);
                cands.push((normalize(sub(sh, ank)), 4f64.to_radians(), "standing"));
            }
            if ctx.torso_ok {
                let s = if ctx.sitting_detected { 14.0 } else { 8.0 };
                cands.push((normalize(sub(sh, hip)), f64::to_radians(s), "torso"));
            }
        }
        let left = mid(sub(j[L_SH], j[R_SH]), sub(j[L_HIP], j[R_HIP]));
        body_fwd = Some(left);
    }
    let mut face_fwd = None;
    if ctx.face_ok {
        if let Some(fe) = &ctx.face {
            let s = if inp.upright { 12.0 } else { 20.0 };
            cands.push((col(&fe.axes, 1), f64::to_radians(s), "face"));
            face_fwd = Some(col(&fe.axes, 2));
        }
    }
    let mut g = fuse_gravity(&cands, 10f64.to_radians());
    if pt.grav_rot != [0.0; 3] {
        g.u = normalize(mat_vec(&rodrigues(pt.grav_rot), g.u));
    }
    let u = g.u;
    let body_fwd = body_fwd.map(|left| normalize(cross(left, u)));
    let facing = body_fwd.or(face_fwd).unwrap_or([0.0, 0.0, -1.0]);

    // Ground
    let (ground_h, ground_seen) = match &joints {
        Some(j) => {
            let gh = FEET.iter().map(|&i| dot(u, j[i])).fold(f64::INFINITY, f64::min);
            let seen = FEET.iter().any(|&i| ctx.weights[i] > 0.5);
            (gh, seen)
        }
        None => {
            let sh = inp.subject_h.unwrap_or(DEFAULT_SUBJECT_H);
            (dot(u, e) - (sh - EYE_TO_TOP_M), false)
        }
    };
    let m = metrics(e, if ground_seen { Some(ground_h) } else { None }, u, facing, inp.mirrored);
    Some(Solved {
        m,
        e,
        ground_h,
        ground_seen,
        u,
        facing,
        face_fwd,
        body_fwd,
        joints,
        fit,
        face_z,
        scale: scale_used,
    })
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let i = pos.floor() as usize;
    let j = (i + 1).min(sorted.len() - 1);
    let fr = pos - i as f64;
    sorted[i] * (1.0 - fr) + sorted[j] * fr
}

fn range_json(v: f64, samples: &mut Vec<f64>, deg: bool) -> String {
    let k = if deg { 180.0 / std::f64::consts::PI } else { 1.0 };
    if !v.is_finite() {
        return "null".into();
    }
    samples.retain(|x| x.is_finite());
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (lo, hi) = if samples.len() >= 4 {
        (percentile(samples, 0.1).min(v), percentile(samples, 0.9).max(v))
    } else {
        (v, v)
    };
    format!("{{\"v\":{},\"lo\":{},\"hi\":{}}}", fmt_num(v * k), fmt_num(lo * k), fmt_num(hi * k))
}

fn line_json(l: [f64; 3]) -> String {
    format!("[{},{},{}]", fmt_num(l[0]), fmt_num(l[1]), fmt_num(l[2]))
}

pub fn analyze(inp_in: &Input) -> String {
    let mut inp_owned = inp_in.clone();
    let err = |e: &str| {
        let mut j = Json::obj();
        j.bool("ok", false).str("error", e);
        j.end()
    };
    if !(inp_in.w > 0.0 && inp_in.h > 0.0 && inp_in.f > 0.0) {
        return err("bad_input");
    }
    // --- focal estimate (plan Stage 1, option 5): only without EXIF and when the user
    // has not picked a lens. Trusted only when the error curve has a clear minimum.
    let mut focal = FocalInfo { f_input: inp_in.f, est: None };
    if inp_in.allow_estimate && inp_in.no_exif {
        if let Some(e) = estimate_focal(inp_in) {
            inp_owned.f = e.f;
            inp_owned.f_sigma = e.sigma_rel;
            focal.est = Some(e);
        }
    }
    let inp = &inp_owned;
    let cam = Cam { f: inp.f, cx: inp.w / 2.0, cy: inp.h / 2.0 };

    // --- static context
    let weights = inp.pose.as_ref().map(|p| pose_weights(p, inp.w, inp.h)).unwrap_or_default();
    let sigma_px = inp.pose.as_ref().map(|p| landmark_sigma_px(p, &weights)).unwrap_or(2.0);
    // legs out of frame => the model's leg lengths are guesses, so body scale is weak
    let legs_seen = !weights.is_empty()
        && weights[L_KNEE] > 0.0
        && weights[R_KNEE] > 0.0
        && (weights[L_ANK] > 0.0 || weights[R_ANK] > 0.0);
    let (base_scale, scale_sigma, height_source, h_est) = match &inp.pose {
        Some(p) => {
            let he = world_height(&p.world);
            let (sg, sa) = if legs_seen { (0.04, 0.07) } else { (0.12, 0.15) };
            match inp.subject_h {
                Some(h) if he > 0.3 => (h / he, sg, "given", he),
                _ if he > 0.3 => (DEFAULT_SUBJECT_H / he, sa, "assumed", he),
                _ => (1.0, sa, "assumed", he),
            }
        }
        None => (1.0, 0.07, if inp.subject_h.is_some() { "given" } else { "assumed" }, f64::NAN),
    };
    let face = inp.face.as_ref().and_then(|f| face_estimate(f, &cam));
    // Face must belong to the same person as the body (eyes close to the pose eyes)
    let mut face_ok = face.is_some();
    if let (Some(fe), Some(p)) = (&face, &inp.pose) {
        let pe = mid2(p.uv[L_EYE], p.uv[R_EYE]);
        let d = ((pe[0] - fe.eye_px[0]).powi(2) + (pe[1] - fe.eye_px[1]).powi(2)).sqrt();
        if d > 0.6 * fe.bbox_h.max(10.0) {
            face_ok = false;
        }
    }

    // posture checks (on the nominal fit, below), need joints first
    let mut ctx = Ctx {
        inp,
        weights,
        sigma_px,
        base_scale,
        scale_sigma,
        face: face.clone(),
        face_ok,
        standing: false,
        torso_ok: false,
        sitting_detected: false,
    };
    // Gravity cues only from landmarks clearly inside the frame. A margin here would let a
    // joint just past the edge switch a cue on in one copy of a photo and off in another.
    ctx.torso_ok = [L_SH, R_SH, L_HIP, R_HIP].iter().all(|&i| seen_strict(inp, i, 0.5));
    let nominal_pt = Perturb { f_mul: 1.0, s_mul: 1.0, face_mul: 1.0, uv_noise: None, grav_rot: [0.0; 3] };
    let first = match solve_once(&ctx, &nominal_pt, None) {
        Some(s) => s,
        None => {
            if inp.pose.is_none() && inp.face.is_none() {
                return err("no_person");
            }
            return err("solve_failed");
        }
    };
    if let Some(j) = &first.joints {
        let w = &ctx.weights;
        let knee_l = joint_angle(j[L_HIP], j[L_KNEE], j[L_ANK]).to_degrees();
        let knee_r = joint_angle(j[R_HIP], j[R_KNEE], j[R_ANK]).to_degrees();
        let hip_l = joint_angle(j[L_SH], j[L_HIP], j[L_KNEE]).to_degrees();
        let hip_r = joint_angle(j[R_SH], j[R_HIP], j[R_KNEE]).to_degrees();
        let _ = w;
        let ankles_vis = seen_strict(inp, L_ANK, 0.7) && seen_strict(inp, R_ANK, 0.7);
        ctx.standing = ankles_vis && knee_l > 160.0 && knee_r > 160.0;
        let knees_vis = seen_strict(inp, L_KNEE, 0.5) && seen_strict(inp, R_KNEE, 0.5);
        ctx.sitting_detected = knees_vis && (knee_l.min(knee_r) < 130.0 || hip_l.min(hip_r) < 130.0);
    }
    let nom = if ctx.standing || ctx.sitting_detected {
        match solve_once(&ctx, &nominal_pt, None) {
            Some(s) => s,
            None => first,
        }
    } else {
        first
    };

    // --- Monte Carlo
    let mut rng = Rng::new(inp.seed.max(1));
    let n_mc = inp.mc;
    let face_sig = ctx.face.as_ref().map_or(0.0, |f| f.z_sigma_rel);
    let (grav_sigma, grav_sources, grav_spread) = {
        // recompute fused sigma for reporting
        let mut srcs = vec![];
        let mut wsum = 0.0;
        if inp.upright && ctx.standing {
            srcs.push("standing");
            wsum += 1.0 / 4f64.to_radians().powi(2);
        }
        if let Some(_) = &nom.joints {
            if inp.upright && ctx.torso_ok {
                srcs.push("torso");
                wsum += 1.0 / f64::to_radians(if ctx.sitting_detected { 14.0 } else { 8.0 }).powi(2);
            }
        }
        if ctx.face_ok && ctx.face.is_some() {
            srcs.push("face");
            wsum += 1.0 / f64::to_radians(if inp.upright { 12.0 } else { 20.0 }).powi(2);
        }
        if srcs.is_empty() {
            (15f64.to_radians(), vec!["assumed_level"], 0.0)
        } else {
            let mut spread: f64 = 0.0;
            if let Some(j) = &nom.joints {
                let mut dirs: Vec<V3> = vec![];
                let sh = mid(j[L_SH], j[R_SH]);
                if srcs.contains(&"standing") {
                    dirs.push(sub(sh, mid(j[L_ANK], j[R_ANK])));
                }
                if srcs.contains(&"torso") {
                    dirs.push(sub(sh, mid(j[L_HIP], j[R_HIP])));
                }
                if let (true, Some(fe)) = (srcs.contains(&"face"), &ctx.face) {
                    dirs.push(col(&fe.axes, 1));
                }
                for a in 0..dirs.len() {
                    for b in a + 1..dirs.len() {
                        spread = spread.max(angle(dirs[a], dirs[b]));
                    }
                }
            }
            ((1.0 / wsum).sqrt(), srcs, spread)
        }
    };
    let mut samples: Vec<Metrics> = Vec::with_capacity(n_mc);
    for _ in 0..n_mc {
        let uv_noise = inp.pose.as_ref().map(|_| {
            (0..N_POSE).map(|_| [rng.normal() * ctx.sigma_px, rng.normal() * ctx.sigma_px]).collect::<Vec<_>>()
        });
        let pt = Perturb {
            f_mul: (1.0 + rng.normal() * inp.f_sigma).max(0.3),
            s_mul: (1.0 + rng.normal() * ctx.scale_sigma).max(0.5),
            face_mul: (1.0 + rng.normal() * face_sig).max(0.5),
            uv_noise,
            grav_rot: [rng.normal() * grav_sigma, 0.0, rng.normal() * grav_sigma],
        };
        if let Some(s) = solve_once(&ctx, &pt, nom.fit.as_ref()) {
            samples.push(s.m);
        }
    }

    // --- assemble JSON
    let m = nom.m;
    let get = |f: fn(&Metrics) -> f64| -> Vec<f64> { samples.iter().map(f).collect() };
    let mut az_s: Vec<f64> = samples.iter().map(|s| m.azim + wrap_pi(s.azim - m.azim)).collect();
    let mut cj = Json::obj();
    cj.raw("pitch_deg", &range_json(m.pitch, &mut get(|s| s.pitch), true))
        .raw("roll_deg", &range_json(m.roll, &mut get(|s| s.roll), true))
        .raw("height_above_eye_m", &range_json(m.h_eye, &mut get(|s| s.h_eye), false))
        .raw("height_above_ground_m", &range_json(m.h_ground, &mut get(|s| s.h_ground), false))
        .raw("distance_to_eyes_m", &range_json(m.dist, &mut get(|s| s.dist), false))
        .raw("horizontal_distance_m", &range_json(m.d_h, &mut get(|s| s.d_h), false))
        .raw("elevation_deg", &range_json(m.elev, &mut get(|s| s.elev), true))
        .raw("azimuth_deg", &range_json(m.azim, &mut az_s, true));
    let camera_json = cj.end();

    // labels
    let framing = framing_label(inp, &ctx, nom.joints.is_some());
    let mut lj = Json::obj();
    lj.str("angle", angle_label(m.elev.to_degrees()))
        .bool("dutch", m.roll.to_degrees().abs() > 5.0)
        .str("side", side_label(m.azim.to_degrees()))
        .str("framing", framing)
        .bool("selfie_distortion", m.dist < 0.6);
    let labels_json = lj.end();

    // subject
    let head_yaw = match (nom.face_fwd, nom.body_fwd) {
        (Some(ff), Some(bf)) => {
            let zf = normalize(reject(ff, nom.u));
            let zb = normalize(reject(bf, nom.u));
            Some(dot(cross(zb, zf), nom.u).atan2(dot(zb, zf)).to_degrees())
        }
        _ => None,
    };
    // height implied by the final scale (differs from the input when the iris depth moved it)
    let (subject_h, height_source) = if h_est.is_finite() && nom.joints.is_some() && legs_seen {
        let h = h_est * nom.scale;
        let given = inp.subject_h.unwrap_or(DEFAULT_SUBJECT_H);
        if (h / given - 1.0).abs() > 0.02 { (h, "body+iris") } else { (given, height_source) }
    } else {
        (inp.subject_h.unwrap_or(DEFAULT_SUBJECT_H), height_source)
    };
    let mut sj = Json::obj();
    sj.num("height_m", subject_h).str("height_source", height_source).opt_num("head_yaw_vs_body_deg", head_yaw);
    let subject_json = sj.end();

    let mut gj = Json::obj();
    gj.strs("sources", &grav_sources)
        .num("spread_deg", grav_spread.to_degrees())
        .num("sigma_deg", grav_sigma.to_degrees());
    let grav_json = gj.end();

    let mut fj = Json::obj();
    fj.opt_num("reproj_rmse_px", nom.fit.as_ref().map(|f| f.rmse_px))
        .num("landmarks_used", nom.fit.as_ref().map_or(0, |f| f.n) as f64)
        .num("landmark_sigma_px", ctx.sigma_px)
        .opt_num("iris_px", ctx.face.as_ref().filter(|_| ctx.face_ok).map(|f| f.iris_px))
        .opt_num("face_depth_m", nom.face_z)
        .bool("iris_used", ctx.face_ok && ctx.face.as_ref().map_or(false, |f| f.used_iris))
        .num("mc_samples", samples.len() as f64);
    let fit_json = fj.end();

    // --- scene (subject frame W)
    let u = nom.u;
    let zw = normalize(reject(nom.facing, u));
    let xw = cross(u, zw);
    let r_wc: M3 = [xw, u, zw];
    let anchor = match &nom.joints {
        Some(j) => mid(j[L_HIP], j[R_HIP]),
        None => nom.e,
    };
    let o_c = add(anchor, scale(u, nom.ground_h - dot(u, anchor)));
    let to_w = |p: V3| mat_vec(&r_wc, sub(p, o_c));
    let cam_w = to_w([0.0; 3]);
    let eye_w = to_w(nom.e);
    let mut scene = Json::obj();
    scene
        .nums("R_wc", &[r_wc[0][0], r_wc[0][1], r_wc[0][2], r_wc[1][0], r_wc[1][1], r_wc[1][2], r_wc[2][0], r_wc[2][1], r_wc[2][2]])
        .nums("o_c", &o_c)
        .nums("cam_pos_w", &cam_w)
        .nums("eye_w", &eye_w)
        .bool("ground_seen", nom.ground_seen);
    if let Some(j) = &nom.joints {
        let flat: Vec<f64> = j.iter().flat_map(|p| to_w(*p)).collect();
        scene.nums("joints_w", &flat);
        let vis: Vec<f64> = ctx.weights.clone();
        scene.nums("joint_w8", &vis);
    }
    if let (true, Some(fpts), Some(fe), Some(fz)) = (ctx.face_ok, &inp.face, &ctx.face, nom.face_z) {
        // back-project the detected landmarks at the solved depth (keeps expression)
        let z_eye = if nom.joints.is_some() { nom.e[2] } else { fz };
        let m_per_px = z_eye / cam.f;
        let flat: Vec<f64> = fpts
            .iter()
            .flat_map(|p| {
                let d = z_eye + (p[2] - fe.z_eyes_px) * m_per_px;
                to_w(scale(cam.ray(p[0], p[1]), d))
            })
            .collect();
        scene.nums("face_w", &flat);
    }
    let scene_json = scene.end();

    // --- overlay
    let horizon = [u[0] / cam.f, u[1] / cam.f, -cam.cx * u[0] / cam.f - cam.cy * u[1] / cam.f + u[2]];
    let ep = cam.project(nom.e);
    let eye_line = [horizon[0], horizon[1], -(horizon[0] * ep[0] + horizon[1] * ep[1])];
    let mut ov = Json::obj();
    ov.raw("horizon", &line_json(horizon)).raw("eye_line", &line_json(eye_line)).nums("eye_px", &ep);
    if let Some(j) = &nom.joints {
        let flat: Vec<f64> = j.iter().flat_map(|p| cam.project(*p)).collect();
        ov.nums("reproj_px", &flat);
    }
    let overlay_json = ov.end();

    // --- warnings
    let mut warn: Vec<&str> = vec![];
    if inp.no_exif {
        warn.push("no_exif");
    }
    if inp.cropped {
        warn.push("cropped");
    }
    if !inp.upright {
        warn.push("not_upright");
    } else if ctx.sitting_detected {
        warn.push("sitting_detected");
    }
    if grav_spread.to_degrees() > 10.0 {
        warn.push("gravity_disagreement");
    }
    if grav_sources.first() == Some(&"assumed_level") {
        warn.push("no_gravity_cue");
    }
    if ctx.face_ok && ctx.face.as_ref().map_or(false, |f| !f.used_iris) {
        warn.push("face_small_no_iris");
    }
    if let Some(f) = &nom.fit {
        if f.rmse_px > 3.0 * ctx.sigma_px {
            warn.push("poor_fit");
        }
    }

    let mode = match (nom.joints.is_some(), ctx.face_ok && nom.face_z.is_some()) {
        (true, true) => "body+face",
        (true, false) => "body",
        _ => "face",
    };
    let mut out = Json::obj();
    out.bool("ok", true)
        .num("version", 1.0)
        .str("mode", mode)
        .raw("camera", &camera_json)
        .raw("labels", &labels_json)
        .raw("subject", &subject_json)
        .raw("gravity", &grav_json)
        .raw("fit", &fit_json)
        .raw("focal", &focal.json(inp.f))
        .raw("scene", &scene_json)
        .raw("overlay", &overlay_json)
        .strs("warnings", &warn);
    out.end()
}

fn mid2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
}

fn framing_label(inp: &Input, ctx: &Ctx, has_body: bool) -> &'static str {
    let h = inp.h;
    if has_body {
        let p = inp.pose.as_ref().unwrap();
        let w = &ctx.weights;
        let seen = |i: usize| w[i] > 0.5 && p.uv[i][1] <= h && p.uv[i][1] >= 0.0;
        let feet = FEET.iter().any(|&i| seen(i));
        if feet {
            let top = p.uv.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min);
            let bot = FEET.iter().map(|&i| p.uv[i][1]).fold(f64::NEG_INFINITY, f64::max);
            return if (bot - top) < 0.5 * h { "long" } else { "full" };
        }
        if seen(L_KNEE) || seen(R_KNEE) {
            return "medium_long";
        }
        if seen(L_HIP) || seen(R_HIP) {
            return "medium";
        }
        if seen(L_SH) || seen(R_SH) {
            let sh = mid2(p.uv[L_SH], p.uv[R_SH])[1];
            let hip = mid2(p.uv[L_HIP], p.uv[R_HIP])[1];
            let frac = (h - sh) / (hip - sh).max(1.0);
            return if frac >= 0.5 { "medium_close_up" } else { "close_up" };
        }
    }
    if let Some(fe) = &ctx.face {
        if fe.bbox_h > 0.55 * h {
            return "extreme_close_up";
        }
        return "close_up";
    }
    if has_body && inp.pose.as_ref().map_or(false, |p| p.vis[NOSE] > 0.5) {
        return "close_up";
    }
    "unknown"
}

// ---- focal length estimate ------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FocalEst {
    pub f: f64,
    pub lo: f64,
    pub hi: f64,
    pub sigma_rel: f64,
    pub rmse_input: f64,
    pub rmse_best: f64,
}

struct FocalInfo {
    f_input: f64,
    est: Option<FocalEst>,
}
impl FocalInfo {
    fn json(&self, f_used: f64) -> String {
        let mut j = Json::obj();
        j.num("f_px", f_used).num("f_input_px", self.f_input);
        match &self.est {
            Some(e) => {
                j.str("source", "estimated")
                    .num("lo_px", e.lo)
                    .num("hi_px", e.hi)
                    .num("rmse_input_px", e.rmse_input)
                    .num("rmse_best_px", e.rmse_best);
            }
            None => {
                j.str("source", "input");
            }
        }
        j.end()
    }
}

fn body_rmse_at(inp: &Input, w: &[f64], sigma_px: f64, f: f64) -> f64 {
    let pose = match &inp.pose {
        Some(p) => p,
        None => return f64::INFINITY,
    };
    let cam = Cam { f, cx: inp.w / 2.0, cy: inp.h / 2.0 };
    let model: Vec<V3> = pose.world.to_vec();
    body_fit(&model, &pose.uv, w, &cam, sigma_px).map_or(f64::INFINITY, |b| b.rmse_px)
}

/// 1-D search over focal length minimising the body reprojection error.
pub fn estimate_focal(inp: &Input) -> Option<FocalEst> {
    let pose = inp.pose.as_ref()?;
    let w = pose_weights(pose, inp.w, inp.h);
    // Perspective evidence needs body depth range: hips and at least one knee in frame,
    // plenty of confident landmarks (plan: "close full-body or wide shots" only).
    let confident = (0..N_POSE).filter(|&i| w[i] > 0.6).count();
    let legs = w[L_HIP] > 0.5 && w[R_HIP] > 0.5 && (w[L_KNEE] > 0.5 || w[R_KNEE] > 0.5);
    if confident < 14 || !legs {
        return None;
    }
    let sigma_px = landmark_sigma_px(pose, &w);
    // multipliers of the input focal: ~12 mm .. ~400 mm for a 26 mm preset
    let n = 26;
    let (m0, m1) = (0.45f64, 15.0f64);
    let ms: Vec<f64> = (0..n).map(|i| m0 * (m1 / m0).powf(i as f64 / (n - 1) as f64)).collect();
    let errs: Vec<f64> = ms.iter().map(|m| body_rmse_at(inp, &w, sigma_px, inp.f * m)).collect();
    let rmse_input = body_rmse_at(inp, &w, sigma_px, inp.f);
    let (bi, _) = errs.iter().enumerate().fold((0, f64::INFINITY), |acc, (i, &e)| if e < acc.1 { (i, e) } else { acc });
    // golden-section refine in log space between neighbours
    let (mut a, mut b) = (ms[bi.saturating_sub(1)].ln(), ms[(bi + 1).min(n - 1)].ln());
    let g = 0.618_033_988_75;
    let fe = |lm: f64| body_rmse_at(inp, &w, sigma_px, inp.f * lm.exp());
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (fe(c), fe(d));
    for _ in 0..14 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = fe(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = fe(d);
        }
    }
    let (lm_best, rmse_best) = if fc < fd { (c, fc) } else { (d, fd) };
    if !rmse_best.is_finite() {
        return None;
    }
    // Clear-minimum test: the preset must fit clearly worse than the best focal,
    // and the misfit must be well above landmark noise.
    // Real photos keep a structural error floor (pose model vs. real body), so measure the
    // preset's *extra* error above the best fit, in quadrature.
    let excess = (rmse_input * rmse_input - rmse_best * rmse_best).max(0.0).sqrt();
    let clear = (rmse_best < 0.65 * rmse_input && rmse_input > 0.5 * sigma_px) || excess > sigma_px;
    // the model must actually fit at the best focal, and the minimum must be inside the range
    let good_fit = rmse_best < 2.5 * sigma_px;
    let interior = bi > 0 && bi < n - 1;
    if !(clear && good_fit && interior) {
        return None;
    }
    // Interval of focal lengths that fit nearly as well (extra error below half the landmark noise)
    let tol = (rmse_best * rmse_best + (0.5 * sigma_px).powi(2)).sqrt();
    let mut lo = lm_best.exp();
    let mut hi = lo;
    for (i, &e) in errs.iter().enumerate() {
        if e <= tol {
            lo = lo.min(ms[i]);
            hi = hi.max(ms[i]);
        }
    }
    let best = lm_best.exp();
    let sigma_rel = ((hi / lo).ln() / 2.56).clamp(0.12, 0.6);
    Some(FocalEst { f: inp.f * best, lo: inp.f * lo, hi: inp.f * hi, sigma_rel, rmse_input, rmse_best })
}

/// Landmark is confidently visible and inside the image (no margin).
fn seen_strict(inp: &Input, i: usize, min_vis: f64) -> bool {
    inp.pose.as_ref().map_or(false, |p| {
        let [u, v] = p.uv[i];
        p.vis[i] > min_vis && u >= 0.0 && v >= 0.0 && u <= inp.w && v <= inp.h
    })
}
