//! CamPose core: zero-dependency, no I/O. Builds natively and for wasm32.
//!
//! # Conventions
//! - Camera frame **C** (OpenCV): x right, y down, z forward. Camera at the origin.
//! - Subject frame **W**: origin on the ground under the mid-hip (body) or under the
//!   eyes (face only). Y = up (against gravity), Z = subject's facing direction made
//!   horizontal, X = Y × Z (the subject's left). Meters.
//! - `U` = up unit vector written in C. pitch = asin(-U·ẑ) (+ = looking down),
//!   roll = atan2(U_x, -U_y). Angles are radians internally, degrees in JSON.
//! - Azimuth α: + = camera on the subject's left.

pub mod exif;
pub mod intrinsics;
pub mod json;
pub mod linalg;
pub mod solve;

/// Analyze a flat input buffer (layout in `solve::parse_input`). Returns result JSON.
pub fn analyze_buffer(buf: &[f64]) -> String {
    match solve::parse_input(buf) {
        Some(inp) => solve::analyze(&inp),
        None => "{\"ok\":false,\"error\":\"bad_input_len\"}".to_string(),
    }
}

/// Intrinsics from raw file bytes (EXIF) and the decoded, orientation-corrected size.
pub fn intrinsics_json(file_bytes: &[u8], w: u32, h: u32, preset_f35: f64, force_preset: bool) -> String {
    let ex = exif::parse(file_bytes);
    let i = intrinsics::compute(ex.as_ref(), w, h, preset_f35, force_preset);
    intrinsics::to_json(&i, w, h)
}
