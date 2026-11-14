//! Stage 1: focal length in pixels from EXIF or a lens preset.

use crate::exif::Exif;
use crate::json::Json;

#[derive(Debug, Clone, PartialEq)]
pub struct Intrinsics {
    /// Focal length in pixels at the decoded (orientation-corrected) image size.
    pub f_px: f64,
    pub f35_mm: f64,
    pub source: &'static str,
    /// Relative 1σ of f_px.
    pub sigma_rel: f64,
    pub device: Option<String>,
    pub lens: Option<String>,
    pub cropped: bool,
    pub front_camera: bool,
    pub has_exif: bool,
}

/// Long-side constant of a 35mm-equivalent focal, given the native aspect ratio.
/// The 35mm equivalent is defined on the diagonal (43.27 mm).
fn long_side_constant(aspect_long_over_short: f64) -> f64 {
    let diag = (36.0f64 * 36.0 + 24.0 * 24.0).sqrt();
    let a = aspect_long_over_short.max(1.0);
    diag * a / (1.0 + a * a).sqrt()
}

/// `w`, `h`: decoded size after EXIF orientation. `preset_f35`: fallback lens.
/// `force_preset`: ignore EXIF and use the preset.
pub fn compute(ex: Option<&Exif>, w: u32, h: u32, preset_f35: f64, force_preset: bool) -> Intrinsics {
    let long = w.max(h) as f64;
    let short = w.min(h).max(1) as f64;
    let decoded_aspect = long / short;
    let mut out = Intrinsics {
        f_px: preset_f35 * long / long_side_constant(4.0 / 3.0),
        f35_mm: preset_f35,
        source: "preset",
        sigma_rel: 0.10,
        device: None,
        lens: None,
        cropped: false,
        front_camera: false,
        has_exif: false,
    };
    let ex = match ex {
        Some(e) => e,
        None => return out,
    };
    out.device = match (&ex.make, &ex.model) {
        (Some(a), Some(b)) if b.starts_with(a.as_str()) => Some(b.clone()),
        (Some(a), Some(b)) => Some(format!("{} {}", a, b)),
        (None, Some(b)) => Some(b.clone()),
        (Some(a), None) => Some(a.clone()),
        _ => None,
    };
    out.lens = ex.lens_model.clone();
    out.front_camera = ex.lens_model.as_deref().map_or(false, |l| l.to_ascii_lowercase().contains("front"));
    out.has_exif = ex.has_focal();

    // native size and crop detection
    let (nat_long, nat_short) = match (ex.pixel_x, ex.pixel_y) {
        (Some(x), Some(y)) if x > 0 && y > 0 => (x.max(y) as f64, x.min(y) as f64),
        _ => (long, short),
    };
    let nat_aspect = nat_long / nat_short;
    if (nat_aspect - decoded_aspect).abs() / nat_aspect > 0.02 {
        out.cropped = true;
    }
    if force_preset {
        out.sigma_rel = if out.cropped { 0.15 } else { 0.10 };
        return out;
    }
    let crop_sigma = if out.cropped { 0.05 } else { 0.0 };

    // 1) focal plane resolution
    if let (Some(fmm), Some(xres), Some(unit)) = (ex.focal_mm, ex.fp_x_res, ex.fp_unit) {
        let per_mm = match unit {
            2 => xres / 25.4,
            3 => xres / 10.0,
            4 => xres,
            _ => 0.0,
        };
        if fmm > 0.0 && per_mm > 0.0 && ex.pixel_x.is_some() {
            let f_native = fmm * per_mm; // pixels at native resolution (sensor x = long side)
            out.f_px = f_native * long / nat_long;
            out.f35_mm = out.f_px * long_side_constant(nat_aspect) / long;
            out.source = "exif_focal_plane";
            out.sigma_rel = 0.02 + crop_sigma;
            return out;
        }
    }
    // 2) 35mm equivalent
    if let Some(f35) = ex.focal_35mm {
        if f35 > 0.0 {
            // Crop is assumed to keep the long side.
            out.f_px = f35 * long / long_side_constant(nat_aspect);
            out.f35_mm = f35;
            out.source = "exif_35mm";
            out.sigma_rel = 0.02 + crop_sigma;
            return out;
        }
    }
    // 3) physical mm only: rough sensor-width guess by device class
    if let Some(fmm) = ex.focal_mm {
        if fmm > 0.0 {
            let make = ex.make.as_deref().unwrap_or("").to_ascii_lowercase();
            let sensor_long_mm = if fmm < 10.0 {
                // phone-sized optics
                7.0
            } else if make.contains("canon") || make.contains("nikon") || make.contains("sony") || make.contains("fuji") {
                23.5
            } else {
                36.0
            };
            out.f_px = fmm / sensor_long_mm * long;
            out.f35_mm = out.f_px * long_side_constant(nat_aspect) / long;
            out.source = "exif_mm_guess";
            out.sigma_rel = 0.12 + crop_sigma;
            return out;
        }
    }
    out
}

pub fn to_json(i: &Intrinsics, w: u32, h: u32) -> String {
    let mut j = Json::obj();
    let hfov = 2.0 * ((w as f64 / 2.0) / i.f_px).atan();
    let vfov = 2.0 * ((h as f64 / 2.0) / i.f_px).atan();
    j.num("f_px", i.f_px)
        .num("f35_mm", i.f35_mm)
        .num("hfov_deg", hfov.to_degrees())
        .num("vfov_deg", vfov.to_degrees())
        .num("sigma_rel", i.sigma_rel)
        .str("source", i.source)
        .opt_str("device", i.device.as_deref())
        .opt_str("lens", i.lens.as_deref())
        .bool("cropped", i.cropped)
        .bool("front_camera", i.front_camera)
        .bool("has_exif", i.has_exif)
        .num("w", w as f64)
        .num("h", h as f64);
    j.end()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exif;

    #[test]
    fn f35_phone() {
        let j = exif::tests::make_jpeg(Some(26), None, (4032, 3024), "back camera");
        let e = exif::parse(&j).unwrap();
        let i = compute(Some(&e), 3024, 4032, 26.0, false);
        assert_eq!(i.source, "exif_35mm");
        // 26 * 4032 / 34.62
        assert!((i.f_px - 26.0 * 4032.0 / long_side_constant(4.0 / 3.0)).abs() < 1e-6);
        assert!((long_side_constant(4.0 / 3.0) - 34.62).abs() < 0.01);
        assert!((long_side_constant(1.5) - 36.0).abs() < 0.01);
        assert!(!i.cropped);
    }

    #[test]
    fn resized_image_scales() {
        let j = exif::tests::make_jpeg(Some(26), None, (4032, 3024), "");
        let e = exif::parse(&j).unwrap();
        let full = compute(Some(&e), 4032, 3024, 26.0, false);
        let half = compute(Some(&e), 2016, 1512, 26.0, false);
        assert!((full.f_px / half.f_px - 2.0).abs() < 1e-9);
    }

    #[test]
    fn crop_detected_and_front() {
        let j = exif::tests::make_jpeg(Some(24), None, (4032, 3024), "iPhone front camera");
        let e = exif::parse(&j).unwrap();
        let i = compute(Some(&e), 3024, 3024, 26.0, false);
        assert!(i.cropped);
        assert!(i.front_camera);
    }

    #[test]
    fn no_exif_uses_preset() {
        let i = compute(None, 1000, 750, 52.0, false);
        assert_eq!(i.source, "preset");
        assert!((i.sigma_rel - 0.10).abs() < 1e-12);
    }
}
