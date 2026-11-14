//! Layer-1 synthetic geometry tests: known camera + synthetic subject -> project -> solve.

use campose_core::linalg::*;
use campose_core::solve::*;

/// Standing synthetic skeleton in subject frame W (Y up, X = subject's left, Z = forward).
fn skeleton_w() -> [V3; 33] {
    let l: [(usize, V3); 17] = [
        (1, [0.015, 1.59, 0.08]),
        (2, [0.032, 1.59, 0.075]),
        (3, [0.045, 1.59, 0.07]),
        (7, [0.075, 1.57, 0.0]),
        (9, [0.025, 1.50, 0.085]),
        (11, [0.19, 1.42, 0.0]),
        (13, [0.22, 1.12, 0.0]),
        (15, [0.23, 0.85, 0.02]),
        (17, [0.235, 0.78, 0.03]),
        (19, [0.23, 0.77, 0.04]),
        (21, [0.21, 0.80, 0.05]),
        (23, [0.10, 0.95, 0.0]),
        (25, [0.10, 0.50, 0.02]),
        (27, [0.10, 0.08, 0.0]),
        (29, [0.10, 0.03, -0.05]),
        (31, [0.10, 0.0, 0.15]),
        (0, [0.0, 1.55, 0.10]),
    ];
    let mut s = [[0.0; 3]; 33];
    for (i, p) in l.iter() {
        s[*i] = *p;
        if *i != 0 {
            // right side partner: left indices 1,2,3 -> 4,5,6; 7->8; 9->10; 11..31 odd -> +1
            let j = match *i {
                1 => 4,
                2 => 5,
                3 => 6,
                k => k + 1,
            };
            s[j] = [-p[0], p[1], p[2]];
        }
    }
    s
}

struct Shot {
    w: f64,
    h: f64,
    f: f64,
    cam_w: V3,
    target_w: V3,
    roll: f64,
}

fn r_cw(shot: &Shot) -> M3 {
    let z = normalize(sub(shot.target_w, shot.cam_w));
    let x = normalize(cross(z, [0.0, 1.0, 0.0]));
    let y = cross(z, x);
    let r = [x, y, z];
    // roll around the optical axis
    mat_mul(&rodrigues([0.0, 0.0, shot.roll]), &r)
}

fn build_input(shot: &Shot, mc: f64) -> Vec<f64> {
    let sk = skeleton_w();
    let r = r_cw(shot);
    let hip = mid(sk[23], sk[24]);
    let mut buf = vec![0.0; INPUT_LEN];
    buf[0] = shot.w;
    buf[1] = shot.h;
    buf[2] = shot.f;
    buf[3] = 0.02;
    buf[4] = 0.0;
    buf[5] = 1.0;
    buf[7] = mc;
    buf[8] = 1.0;
    buf[12] = 7.0;
    let cam = Cam { f: shot.f, cx: shot.w / 2.0, cy: shot.h / 2.0 };
    for i in 0..33 {
        let m = mat_vec(&r, sub(sk[i], hip)); // camera-aligned, hip-centred (like BlazePose world)
        let pc = mat_vec(&r, sub(sk[i], shot.cam_w));
        let uv = cam.project(pc);
        buf[OFF_POSE2D + i * 3] = uv[0];
        buf[OFF_POSE2D + i * 3 + 1] = uv[1];
        buf[OFF_POSE2D + i * 3 + 2] = 0.99;
        buf[OFF_WORLD + i * 3] = m[0];
        buf[OFF_WORLD + i * 3 + 1] = m[1];
        buf[OFF_WORLD + i * 3 + 2] = m[2];
    }
    // give the skeleton's own height so the solver's scale is exactly 1
    buf[4] = world_height(&sk.map(|p| sub(p, hip)));
    buf
}

fn get(json: &str, key: &str) -> f64 {
    let pat = format!("\"{}\":{{\"v\":", key);
    let i = json.find(&pat).unwrap_or_else(|| panic!("{} missing in {}", key, json)) + pat.len();
    let rest = &json[i..];
    let end = rest.find(',').unwrap();
    rest[..end].parse().unwrap()
}
fn get_range(json: &str, key: &str) -> (f64, f64, f64) {
    let pat = format!("\"{}\":{{\"v\":", key);
    let i = json.find(&pat).unwrap() + pat.len() - 5;
    let rest = &json[i..];
    let seg = &rest[..rest.find('}').unwrap()];
    let nums: Vec<f64> = seg
        .split(|c| c == ':' || c == ',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    (nums[0], nums[1], nums[2])
}

fn expected(shot: &Shot) -> (f64, f64, f64, f64, f64) {
    let sk = skeleton_w();
    let eye = mid(sk[2], sk[5]);
    let d = sub(eye, shot.cam_w);
    let dh = (d[0] * d[0] + d[2] * d[2]).sqrt();
    let h_eye = shot.cam_w[1] - eye[1];
    let z = normalize(sub(shot.target_w, shot.cam_w));
    let pitch = (-z[1]).asin().to_degrees();
    let az = (-d[0]).atan2(-d[2]).to_degrees();
    (pitch, h_eye, norm(d), dh, az)
}

fn run(shot: &Shot) -> String {
    let buf = build_input(shot, 0.0);
    let out = campose_core::analyze_buffer(&buf);
    assert!(out.contains("\"ok\":true"), "{}", out);
    out
}

#[test]
fn recovers_front_high_angle() {
    let shot = Shot { w: 1200.0, h: 1600.0, f: 1300.0, cam_w: [0.0, 2.2, 4.0], target_w: [0.0, 1.0, 0.0], roll: 0.0 };
    let out = run(&shot);
    let (pitch, h_eye, dist, dh, az) = expected(&shot);
    assert!((get(&out, "pitch_deg") - pitch).abs() < 1e-3, "{}", out);
    assert!((get(&out, "height_above_eye_m") - h_eye).abs() < 1e-3);
    assert!((get(&out, "distance_to_eyes_m") - dist).abs() < 1e-3);
    assert!((get(&out, "horizontal_distance_m") - dh).abs() < 1e-3);
    assert!((get(&out, "height_above_ground_m") - 2.2).abs() < 1e-3);
    assert!((get(&out, "azimuth_deg") - az).abs() < 1e-2);
    assert!(get(&out, "pitch_deg") > 0.0, "looking down => pitch > 0");
    assert!(get(&out, "height_above_eye_m") > 0.0, "camera above eyes => dh > 0");
    assert!(get(&out, "roll_deg").abs() < 1e-3);
    assert!(out.contains("\"angle\":\"high\""));
}

#[test]
fn subject_left_side_is_positive_azimuth() {
    // camera on the subject's left (+X), low angle
    let shot = Shot { w: 1600.0, h: 1200.0, f: 1100.0, cam_w: [2.5, 0.6, 2.5], target_w: [0.0, 1.0, 0.0], roll: 0.0 };
    let out = run(&shot);
    let (pitch, h_eye, dist, _, az) = expected(&shot);
    let got_az = get(&out, "azimuth_deg");
    assert!(got_az > 0.0, "{}", out);
    assert!((got_az - az).abs() < 0.05, "az {} vs {}", got_az, az);
    assert!((get(&out, "pitch_deg") - pitch).abs() < 1e-3);
    assert!(get(&out, "pitch_deg") < 0.0, "looking up => pitch < 0");
    assert!((get(&out, "height_above_eye_m") - h_eye).abs() < 1e-3);
    assert!((get(&out, "distance_to_eyes_m") - dist).abs() < 1e-3);
    assert!(out.contains("\"side\":\"three_quarter_left\""));
}

#[test]
fn subject_right_side_is_negative_and_mirror_flips() {
    let shot = Shot { w: 1600.0, h: 1200.0, f: 1100.0, cam_w: [-3.0, 1.5, 0.5], target_w: [0.0, 1.0, 0.0], roll: 0.0 };
    let out = run(&shot);
    let az = get(&out, "azimuth_deg");
    assert!(az < -60.0, "{}", out);
    let mut buf = build_input(&shot, 0.0);
    buf[6] = 1.0; // mirrored
    let out2 = campose_core::analyze_buffer(&buf);
    assert!((get(&out2, "azimuth_deg") + az).abs() < 1e-6);
}

#[test]
fn roll_sign() {
    // positive roll about optical axis (camera rotated clockwise as seen from behind)
    let shot = Shot { w: 1200.0, h: 1600.0, f: 1300.0, cam_w: [0.0, 1.6, 3.5], target_w: [0.0, 1.0, 0.0], roll: 15f64.to_radians() };
    let out = run(&shot);
    let roll = get(&out, "roll_deg");
    // roll prior shrinks slightly toward 0; sign and rough size must hold
    assert!(roll.abs() > 12.0 && roll.abs() < 15.5, "{}", out);
    assert!(out.contains("\"dutch\":true"));
    // pitch_roll / u_from_pitch_roll are inverse
    let u = u_from_pitch_roll(0.3, -0.2);
    let (p, r) = pitch_roll(u);
    assert!((p - 0.3).abs() < 1e-12 && (r + 0.2).abs() < 1e-12);
}

#[test]
fn monte_carlo_ranges_bracket_value() {
    let shot = Shot { w: 1200.0, h: 1600.0, f: 1300.0, cam_w: [0.5, 1.9, 3.0], target_w: [0.0, 1.1, 0.0], roll: 0.0 };
    let buf = build_input(&shot, 64.0);
    let out = campose_core::analyze_buffer(&buf);
    for k in ["pitch_deg", "distance_to_eyes_m", "height_above_eye_m", "azimuth_deg"] {
        let (v, lo, hi) = get_range(&out, k);
        assert!(lo <= v && v <= hi && hi > lo, "{}: {} {} {}", k, v, lo, hi);
    }
    let (v, lo, hi) = get_range(&out, "distance_to_eyes_m");
    assert!((hi - lo) / v < 0.4, "distance range too wide");
}

#[test]
fn face_only_depth_from_iris() {
    let (w, h, f) = (1000.0, 1000.0, 900.0);
    let z = 0.5;
    let mut buf = vec![0.0; INPUT_LEN];
    buf[0] = w;
    buf[1] = h;
    buf[2] = f;
    buf[5] = 1.0;
    buf[9] = 1.0;
    let cx = w / 2.0;
    let cy = h / 2.0 - 40.0;
    let ipd = f * IPD_M / z;
    let iris = f * IRIS_DIAM_M / z;
    let face_h = f * 0.20 / z;
    let mut set = |i: usize, u: f64, v: f64| {
        buf[OFF_FACE + i * 3] = u;
        buf[OFF_FACE + i * 3 + 1] = v;
        buf[OFF_FACE + i * 3 + 2] = 0.0;
    };
    for i in 0..478 {
        set(i, cx, cy);
    }
    set(468, cx - ipd / 2.0, cy);
    set(473, cx + ipd / 2.0, cy);
    for (c, base) in [(cx - ipd / 2.0, 469), (cx + ipd / 2.0, 474)] {
        set(base, c + iris / 2.0, cy);
        set(base + 1, c, cy - iris / 2.0);
        set(base + 2, c - iris / 2.0, cy);
        set(base + 3, c, cy + iris / 2.0);
    }
    set(33, cx - ipd * 0.75, cy);
    set(263, cx + ipd * 0.75, cy);
    set(10, cx, cy - face_h * 0.4);
    set(152, cx, cy + face_h * 0.6);
    let out = campose_core::analyze_buffer(&buf);
    assert!(out.contains("\"mode\":\"face\""), "{}", out);
    let ray = [0.0, -40.0 / f, 1.0];
    let d = z * norm(ray);
    assert!((get(&out, "distance_to_eyes_m") - d).abs() < 1e-3, "{}", out);
    assert!(get(&out, "azimuth_deg").abs() < 1.0);
    assert!(out.contains("\"selfie_distortion\":true"));
}

#[test]
fn no_person_is_an_error() {
    let mut buf = vec![0.0; INPUT_LEN];
    buf[0] = 100.0;
    buf[1] = 100.0;
    buf[2] = 100.0;
    let out = campose_core::analyze_buffer(&buf);
    assert!(out.contains("\"ok\":false"));
}

#[test]
fn focal_estimate_recovers_wrong_preset() {
    // true lens is wide (f=800) at close range; the "preset" claims f=1600
    let shot = Shot { w: 1200.0, h: 1600.0, f: 800.0, cam_w: [0.3, 1.4, 1.9], target_w: [0.0, 1.0, 0.0], roll: 0.0 };
    let mut buf = build_input(&shot, 0.0);
    buf[2] = 1600.0;
    buf[10] = 1.0; // no exif
    buf[13] = 1.0; // estimate allowed
    let out = campose_core::analyze_buffer(&buf);
    assert!(out.contains("\"source\":\"estimated\""), "{}", out);
    let i = out.find("\"focal\":{\"f_px\":").unwrap() + 16;
    let f: f64 = out[i..i + out[i..].find(',').unwrap()].parse().unwrap();
    assert!((f / 800.0 - 1.0).abs() < 0.05, "f = {}", f);
    let (_, h_eye, dist, _, _) = expected(&shot);
    assert!((get(&out, "distance_to_eyes_m") - dist).abs() / dist < 0.05);
    assert!((get(&out, "height_above_eye_m") - h_eye).abs() < 0.05);
    // without permission the preset is kept
    buf[13] = 0.0;
    let out2 = campose_core::analyze_buffer(&buf);
    assert!(out2.contains("\"source\":\"input\""));
}


#[test]
fn no_focal_estimate_for_head_and_shoulders() {
    let shot = Shot { w: 1200.0, h: 1600.0, f: 800.0, cam_w: [0.3, 1.4, 1.9], target_w: [0.0, 1.0, 0.0], roll: 0.0 };
    let mut buf = build_input(&shot, 0.0);
    buf[2] = 1600.0;
    buf[10] = 1.0;
    buf[13] = 1.0;
    for i in 23..33 {
        buf[OFF_POSE2D + i * 3 + 2] = 0.05; // hips and legs not visible (selfie crop)
    }
    let out = campose_core::analyze_buffer(&buf);
    assert!(out.contains("\"source\":\"input\""), "{}", out);
}
