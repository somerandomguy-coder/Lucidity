// Human-readable labels and number formatting.

export const ANGLE = {
  eye_level: 'Eye level', high: 'High angle', steep_high: 'Steep high', overhead: 'Overhead',
  low: 'Low angle', steep_low: 'Steep low', worms_eye: "Worm's-eye",
};
export const SIDE = {
  front: 'Front', three_quarter_left: '3/4 from subject’s left', three_quarter_right: '3/4 from subject’s right',
  profile_left: 'Profile (left side)', profile_right: 'Profile (right side)', behind: 'Behind',
};
export const SIDE_SHORT = {
  front: 'Front', three_quarter_left: '3/4 left', three_quarter_right: '3/4 right',
  profile_left: 'Profile L', profile_right: 'Profile R', behind: 'Behind',
};
export const FRAMING = {
  extreme_close_up: 'Extreme close-up', close_up: 'Close-up', medium_close_up: 'Medium close-up',
  medium: 'Medium shot', medium_long: 'Medium long', full: 'Full shot', long: 'Long shot', unknown: 'Unknown framing',
};
export const LENS_SOURCE = {
  exif_35mm: 'EXIF', exif_focal_plane: 'EXIF', exif_mm_guess: 'EXIF (approx.)', preset: 'Preset (assumed)',
  estimated: 'Estimated from body perspective',
};
export const WARN = {
  no_exif: 'No lens info in the file. Distances scale with focal length, so pick the lens under Assumptions if you know it.',
  cropped: 'The photo looks cropped, so the lens centre and focal length may be off.',
  not_upright: 'Marked “not upright”: posture is ignored for tilt, so the angles are less certain.',
  sitting_detected: 'The subject looks seated or bent, so posture-based tilt is less reliable.',
  gravity_disagreement: 'Body and face disagree about which way is up (leaning or head tilt?).',
  no_gravity_cue: 'No posture cue found, so the camera is assumed level.',
  face_small_no_iris: 'The face is too small for the iris check, so distance uses average eye spacing.',
  poor_fit: 'The skeleton fit is poor (unusual pose or wrong person?). Check the overlay.',
};
export const LENS_PRESETS = [
  ['Phone 0.5× ultrawide', 13], ['Phone 1×', 26], ['Phone 2×', 52], ['Phone 3×', 77], ['Phone 5×', 120],
  ['Front / selfie', 24], ['Camera 35 mm', 35], ['Camera 50 mm', 50], ['Camera 85 mm', 85],
];

const sgn = (v, d = 0) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
export const m = (v, d = 2) => (v == null || !isFinite(v) ? '—' : `${v.toFixed(d)} m`);
export const deg = (v, d = 0) => (v == null || !isFinite(v) ? '—' : `${v.toFixed(d)}°`);
export const sm = (v, d = 2) => (v == null || !isFinite(v) ? '—' : `${sgn(v, d)} m`);

export function tilt(pitch) {
  if (pitch == null) return '—';
  const a = Math.abs(pitch);
  if (a < 1) return 'level';
  return `${a.toFixed(0)}° ${pitch > 0 ? 'down' : 'up'}`;
}

export function headline(r) {
  const c = r.camera, l = r.labels;
  const h = c.height_above_eye_m.v;
  const pos = Math.abs(h) < 0.05 ? 'at eye level' : `${Math.abs(h).toFixed(2)} m ${h > 0 ? 'above' : 'below'} eye level`;
  return `Camera ${pos}, tilted ${tilt(c.pitch_deg.v)}, ${c.distance_to_eyes_m.v.toFixed(1)} m away, ` +
    `${(SIDE[l.side] || l.side).replace('3/4', 'three-quarter')}. ${ANGLE[l.angle]}, ${FRAMING[l.framing].toLowerCase()}.`;
}

export function range(x, fmt) {
  if (!x) return '—';
  return `${fmt(x.lo)} … ${fmt(x.hi)}`;
}
