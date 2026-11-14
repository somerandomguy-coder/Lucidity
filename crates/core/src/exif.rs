//! Minimal dependency-free EXIF reader (JPEG APP1, PNG eXIf, WebP EXIF chunk).
//! Reads only the tags the intrinsics stage needs.

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Exif {
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens_model: Option<String>,
    pub orientation: Option<u16>,
    pub focal_mm: Option<f64>,
    pub focal_35mm: Option<f64>,
    pub fp_x_res: Option<f64>,
    pub fp_y_res: Option<f64>,
    pub fp_unit: Option<u16>,
    pub pixel_x: Option<u32>,
    pub pixel_y: Option<u32>,
    pub digital_zoom: Option<f64>,
}

impl Exif {
    pub fn has_focal(&self) -> bool {
        self.focal_35mm.map_or(false, |v| v > 0.0) || self.focal_mm.map_or(false, |v| v > 0.0)
    }
}

/// Find the raw TIFF block ("II*\0" / "MM\0*") inside a JPEG, PNG or WebP file.
pub fn find_tiff(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.len() < 12 {
        return None;
    }
    // JPEG
    if bytes[0] == 0xFF && bytes[1] == 0xD8 {
        let mut i = 2;
        while i + 4 <= bytes.len() {
            if bytes[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = bytes[i + 1];
            if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) || marker == 0xFF {
                i += if marker == 0xFF { 1 } else { 2 };
                continue;
            }
            if marker == 0xDA || marker == 0xD9 {
                return None; // start of scan: no more metadata
            }
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            let seg_start = i + 4;
            let seg_end = (i + 2 + len).min(bytes.len());
            if marker == 0xE1 && seg_end > seg_start + 6 && &bytes[seg_start..seg_start + 6] == b"Exif\0\0" {
                return Some(&bytes[seg_start + 6..seg_end]);
            }
            i = i + 2 + len;
        }
        return None;
    }
    // PNG
    if &bytes[0..8] == b"\x89PNG\r\n\x1a\n" {
        let mut i = 8;
        while i + 8 <= bytes.len() {
            let len = u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
            let ty = &bytes[i + 4..i + 8];
            let start = i + 8;
            let end = start.saturating_add(len).min(bytes.len());
            if ty == b"eXIf" {
                return Some(&bytes[start..end]);
            }
            if ty == b"IDAT" || ty == b"IEND" {
                return None;
            }
            i = end + 4;
        }
        return None;
    }
    // WebP (RIFF)
    if &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        let mut i = 12;
        while i + 8 <= bytes.len() {
            let ty = &bytes[i..i + 4];
            let len = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
            let start = i + 8;
            let end = start.saturating_add(len).min(bytes.len());
            if ty == b"EXIF" {
                let d = &bytes[start..end];
                // some writers keep the "Exif\0\0" prefix
                if d.len() > 6 && &d[0..6] == b"Exif\0\0" {
                    return Some(&d[6..]);
                }
                return Some(d);
            }
            i = end + (len & 1);
        }
    }
    None
}

struct Tiff<'a> {
    d: &'a [u8],
    le: bool,
}

impl<'a> Tiff<'a> {
    fn u16(&self, o: usize) -> Option<u16> {
        let b = self.d.get(o..o + 2)?;
        Some(if self.le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) })
    }
    fn u32(&self, o: usize) -> Option<u32> {
        let b = self.d.get(o..o + 4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Some(if self.le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) })
    }
    /// Offset of the value data of an IFD entry at `e`.
    fn value_off(&self, e: usize, ty: u16, count: u32) -> Option<usize> {
        let size = match ty {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 => 8,
            _ => return None,
        } * count as usize;
        if size <= 4 {
            Some(e + 8)
        } else {
            Some(self.u32(e + 8)? as usize)
        }
    }
    fn num(&self, e: usize) -> Option<f64> {
        let ty = self.u16(e + 2)?;
        let count = self.u32(e + 4)?;
        let o = self.value_off(e, ty, count)?;
        match ty {
            3 => self.u16(o).map(|v| v as f64),
            4 => self.u32(o).map(|v| v as f64),
            5 => {
                let n = self.u32(o)? as f64;
                let d = self.u32(o + 4)? as f64;
                if d == 0.0 { None } else { Some(n / d) }
            }
            10 => {
                let n = self.u32(o)? as i32 as f64;
                let d = self.u32(o + 4)? as i32 as f64;
                if d == 0.0 { None } else { Some(n / d) }
            }
            _ => None,
        }
    }
    fn string(&self, e: usize) -> Option<String> {
        let ty = self.u16(e + 2)?;
        if ty != 2 {
            return None;
        }
        let count = self.u32(e + 4)?;
        let o = self.value_off(e, ty, count)?;
        let raw = self.d.get(o..o + count as usize)?;
        let s: String = raw.iter().take_while(|&&c| c != 0).map(|&c| c as char).collect();
        let s = s.trim().to_string();
        if s.is_empty() { None } else { Some(s) }
    }
}

pub fn parse(bytes: &[u8]) -> Option<Exif> {
    let d = find_tiff(bytes)?;
    if d.len() < 8 {
        return None;
    }
    let le = match &d[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let t = Tiff { d, le };
    if t.u16(2)? != 42 {
        return None;
    }
    let mut ex = Exif::default();
    let ifd0 = t.u32(4)? as usize;
    let mut sub_ifd: Option<usize> = None;
    let read_ifd = |off: usize, ex: &mut Exif, sub: &mut Option<usize>| -> Option<()> {
        let n = t.u16(off)? as usize;
        if n > 1000 {
            return None;
        }
        for k in 0..n {
            let e = off + 2 + k * 12;
            let tag = t.u16(e)?;
            match tag {
                0x010F => ex.make = t.string(e),
                0x0110 => ex.model = t.string(e),
                0x0112 => ex.orientation = t.num(e).map(|v| v as u16),
                0x8769 => *sub = t.num(e).map(|v| v as usize),
                0x920A => ex.focal_mm = t.num(e),
                0xA405 => ex.focal_35mm = t.num(e),
                0xA20E => ex.fp_x_res = t.num(e),
                0xA20F => ex.fp_y_res = t.num(e),
                0xA210 => ex.fp_unit = t.num(e).map(|v| v as u16),
                0xA002 => ex.pixel_x = t.num(e).map(|v| v as u32),
                0xA003 => ex.pixel_y = t.num(e).map(|v| v as u32),
                0xA434 => ex.lens_model = t.string(e),
                0xA404 => ex.digital_zoom = t.num(e),
                _ => {}
            }
        }
        Some(())
    };
    read_ifd(ifd0, &mut ex, &mut sub_ifd);
    if let Some(s) = sub_ifd {
        let mut dummy = None;
        read_ifd(s, &mut ex, &mut dummy);
    }
    Some(ex)
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Build a little-endian JPEG with an EXIF block (test helper, also used by intrinsics tests).
    pub fn make_jpeg(f35: Option<u16>, f_mm: Option<(u32, u32)>, px: (u32, u32), lens: &str) -> Vec<u8> {
        let mut tiff: Vec<u8> = b"II*\0".to_vec();
        tiff.extend(8u32.to_le_bytes());
        // IFD0: Make, ExifIFD pointer
        let make = b"TestCo\0";
        let ifd0_entries = 2u16;
        let ifd0_size = 2 + 12 * ifd0_entries as usize + 4;
        let make_off = 8 + ifd0_size;
        let sub_off = make_off + make.len();
        tiff.extend(ifd0_entries.to_le_bytes());
        tiff.extend(0x010Fu16.to_le_bytes());
        tiff.extend(2u16.to_le_bytes());
        tiff.extend((make.len() as u32).to_le_bytes());
        tiff.extend((make_off as u32).to_le_bytes());
        tiff.extend(0x8769u16.to_le_bytes());
        tiff.extend(4u16.to_le_bytes());
        tiff.extend(1u32.to_le_bytes());
        tiff.extend((sub_off as u32).to_le_bytes());
        tiff.extend(0u32.to_le_bytes());
        tiff.extend(make);
        // sub IFD
        let mut entries: Vec<(u16, u16, u32, Vec<u8>)> = vec![];
        if let Some(v) = f35 {
            let mut b = v.to_le_bytes().to_vec();
            b.extend([0, 0]);
            entries.push((0xA405, 3, 1, b));
        }
        if let Some((n, d)) = f_mm {
            let mut b = n.to_le_bytes().to_vec();
            b.extend(d.to_le_bytes());
            entries.push((0x920A, 5, 1, b));
        }
        entries.push((0xA002, 4, 1, px.0.to_le_bytes().to_vec()));
        entries.push((0xA003, 4, 1, px.1.to_le_bytes().to_vec()));
        let mut lb = lens.as_bytes().to_vec();
        lb.push(0);
        entries.push((0xA434, 2, lb.len() as u32, lb));
        let n = entries.len();
        let data_start = sub_off + 2 + 12 * n + 4;
        let mut data: Vec<u8> = vec![];
        let mut ifd: Vec<u8> = (n as u16).to_le_bytes().to_vec();
        for (tag, ty, count, val) in entries {
            ifd.extend(tag.to_le_bytes());
            ifd.extend(ty.to_le_bytes());
            ifd.extend(count.to_le_bytes());
            if val.len() <= 4 {
                let mut v = val.clone();
                v.resize(4, 0);
                ifd.extend(v);
            } else {
                ifd.extend(((data_start + data.len()) as u32).to_le_bytes());
                data.extend(val);
            }
        }
        ifd.extend(0u32.to_le_bytes());
        tiff.extend(ifd);
        tiff.extend(data);
        let mut jpg = vec![0xFF, 0xD8, 0xFF, 0xE1];
        let len = (tiff.len() + 6 + 2) as u16;
        jpg.extend(len.to_be_bytes());
        jpg.extend(b"Exif\0\0");
        jpg.extend(tiff);
        jpg.extend([0xFF, 0xD9]);
        jpg
    }

    #[test]
    fn parses_synthetic_jpeg() {
        let j = make_jpeg(Some(26), Some((5700, 1000)), (4032, 3024), "iPhone back camera");
        let e = parse(&j).unwrap();
        assert_eq!(e.make.as_deref(), Some("TestCo"));
        assert_eq!(e.focal_35mm, Some(26.0));
        assert!((e.focal_mm.unwrap() - 5.7).abs() < 1e-9);
        assert_eq!(e.pixel_x, Some(4032));
        assert_eq!(e.pixel_y, Some(3024));
        assert_eq!(e.lens_model.as_deref(), Some("iPhone back camera"));
    }

    #[test]
    fn no_exif_returns_none() {
        assert!(parse(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 2, 0, 0, 0, 0, 0, 0]).is_none());
        assert!(parse(b"not an image at all").is_none());
    }
}
