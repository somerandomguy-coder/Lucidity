//! Raw `extern "C"` wasm exports (no wasm-bindgen). JS side: web/core.js.
//!
//! Memory protocol: JS calls `cp_alloc(n)` to get a buffer, writes into it, calls a
//! function, then reads the UTF-8 result at `cp_out_ptr()` with the returned length.

use std::cell::UnsafeCell;

struct Out(UnsafeCell<Vec<u8>>);
unsafe impl Sync for Out {}
static OUT: Out = Out(UnsafeCell::new(Vec::new()));

fn set_out(s: String) -> usize {
    // SAFETY: wasm32 here is single-threaded; JS reads OUT before the next call.
    let v = unsafe { &mut *OUT.0.get() };
    *v = s.into_bytes();
    v.len()
}

#[no_mangle]
pub extern "C" fn cp_alloc(n: usize) -> *mut u8 {
    let mut v: Vec<u8> = Vec::with_capacity(n.max(8));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `p` must come from `cp_alloc(n)` with the same `n`.
#[no_mangle]
pub unsafe extern "C" fn cp_free(p: *mut u8, n: usize) {
    drop(Vec::from_raw_parts(p, 0, n.max(8)));
}

#[no_mangle]
pub extern "C" fn cp_out_ptr() -> *const u8 {
    unsafe { (*OUT.0.get()).as_ptr() }
}

#[no_mangle]
pub extern "C" fn cp_input_len() -> usize {
    campose_core::solve::INPUT_LEN
}

/// # Safety
/// `p` points to `n` f64 values (8-byte aligned).
#[no_mangle]
pub unsafe extern "C" fn cp_analyze(p: *const f64, n: usize) -> usize {
    let buf = std::slice::from_raw_parts(p, n);
    set_out(campose_core::analyze_buffer(buf))
}

/// # Safety
/// `p` points to `n` bytes (may be 0 when no file bytes are available).
#[no_mangle]
pub unsafe extern "C" fn cp_intrinsics(p: *const u8, n: usize, w: u32, h: u32, preset_f35: f64, force: u32) -> usize {
    let bytes: &[u8] = if n == 0 { &[] } else { std::slice::from_raw_parts(p, n) };
    set_out(campose_core::intrinsics_json(bytes, w, h, preset_f35, force != 0))
}

/// Tuning knob: body rotation prior σ in degrees (default 3).
#[no_mangle]
pub extern "C" fn cp_set_rot_prior(deg: f64) {
    campose_core::solve::set_rot_prior_deg(deg);
}
