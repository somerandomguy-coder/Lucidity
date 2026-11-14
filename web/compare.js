// Side-by-side comparison of two analysed photos: overlays, plain-words summary,
// metric table with differences, and side / top-down diagrams of both camera positions.
import * as F from './format.js';

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const COL_A = '#ffb35c', COL_B = '#5ee0d0';

function summary(a, b) {
  const ca = a.result.camera, cb = b.result.camera;
  const parts = [];
  const dh = cb.height_above_eye_m.v - ca.height_above_eye_m.v;
  if (Math.abs(dh) >= 0.05) parts.push(`${Math.abs(dh).toFixed(2)} m ${dh > 0 ? 'higher' : 'lower'} (relative to the eyes)`);
  const dd = cb.distance_to_eyes_m.v - ca.distance_to_eyes_m.v;
  if (Math.abs(dd) >= 0.05) parts.push(`${Math.abs(dd).toFixed(2)} m ${dd > 0 ? 'further away' : 'closer'}`);
  const dp = cb.pitch_deg.v - ca.pitch_deg.v;
  if (Math.abs(dp) >= 2) parts.push(`tilted ${Math.abs(dp).toFixed(0)}° more ${dp > 0 ? 'down' : 'up'}`);
  let daz = cb.azimuth_deg.v - ca.azimuth_deg.v;
  daz = ((daz + 540) % 360) - 180;
  if (Math.abs(daz) >= 5) parts.push(`${Math.abs(daz).toFixed(0)}° further round to the subject’s ${daz > 0 ? 'left' : 'right'}`);
  const df = b.intr.f35_mm / a.intr.f35_mm;
  if (Math.abs(df - 1) > 0.15) parts.push(`with a ${df > 1 ? 'longer' : 'wider'} lens (${Math.round(b.intr.f35_mm)} vs ${Math.round(a.intr.f35_mm)} mm)`);
  if (!parts.length) return 'Both photos were taken from almost the same camera position.';
  const last = parts.pop();
  return `Compared with A, photo B was taken ${parts.length ? parts.join(', ') + ' and ' : ''}${last}.`;
}

/** Side view: horizontal distance vs height relative to the eyes. */
function sideView(a, b) {
  const W = 340, H = 200, ox = 290, eyeY = 100;
  const pts = [a, b].map((p) => ({ d: p.result.camera.horizontal_distance_m.v, h: p.result.camera.height_above_eye_m.v, pitch: p.result.camera.pitch_deg.v }));
  const maxD = Math.max(0.5, ...pts.map((p) => p.d));
  const maxH = Math.max(0.3, ...pts.map((p) => Math.abs(p.h)));
  const s = Math.min(250 / maxD, 80 / maxH);
  const cam = (p, col, label) => {
    const x = ox - p.d * s, y = eyeY - p.h * s;
    const a = p.pitch * Math.PI / 180;
    return `<line x1="${x}" y1="${y}" x2="${ox}" y2="${eyeY}" stroke="${col}" stroke-dasharray="4 3" stroke-width="1.3" opacity=".7"/>
      <line x1="${x}" y1="${y}" x2="${x + Math.cos(a) * 26}" y2="${y + Math.sin(a) * 26}" stroke="${col}" stroke-width="2"/>
      <rect x="${x - 9}" y="${y - 6}" width="18" height="12" rx="3" fill="${col}"/>
      <text x="${x}" y="${y + 4}" text-anchor="middle" font-size="9" font-weight="700" fill="#06121f">${label}</text>`;
  };
  return `<svg viewBox="0 0 ${W} ${H}" class="cmp-svg" role="img" aria-label="Side view of both camera positions">
    <text x="8" y="16" class="cmp-cap">Side view</text>
    <line x1="10" y1="${eyeY}" x2="${W - 10}" y2="${eyeY}" stroke="var(--muted)" stroke-dasharray="2 3"/>
    <text x="10" y="${eyeY - 4}" class="cmp-cap">eye level</text>
    <circle cx="${ox}" cy="${eyeY - 6}" r="8" fill="none" stroke="var(--fg2)" stroke-width="2"/>
    <path d="M${ox} ${eyeY + 2}v40M${ox} ${eyeY + 42}l-8 30M${ox} ${eyeY + 42}l8 30M${ox - 14} ${eyeY + 16}h28" stroke="var(--fg2)" stroke-width="2" fill="none" stroke-linecap="round"/>
    ${cam(pts[0], COL_A, 'A')}${cam(pts[1], COL_B, 'B')}
  </svg>`;
}

/** Top view: subject in the middle facing down; cameras placed by side angle and distance. */
function topView(a, b) {
  const W = 220, H = 200, cx = 110, cy = 92;
  const pts = [a, b].map((p) => ({ d: p.result.camera.horizontal_distance_m.v, az: p.result.camera.azimuth_deg.v * Math.PI / 180 }));
  const s = 78 / Math.max(0.5, ...pts.map((p) => p.d));
  // facing the viewer (down the page); subject's left is screen right
  const pos = (p) => [cx + Math.sin(p.az) * p.d * s, cy + Math.cos(p.az) * p.d * s];
  const cam = (p, col, label) => {
    const [x, y] = pos(p);
    return `<line x1="${x}" y1="${y}" x2="${cx}" y2="${cy}" stroke="${col}" stroke-dasharray="4 3" stroke-width="1.3" opacity=".7"/>
      <circle cx="${x}" cy="${y}" r="9" fill="${col}"/><text x="${x}" y="${y + 3.5}" text-anchor="middle" font-size="10" font-weight="700" fill="#06121f">${label}</text>`;
  };
  return `<svg viewBox="0 0 ${W} ${H}" class="cmp-svg" role="img" aria-label="Top view of both camera positions">
    <text x="8" y="16" class="cmp-cap">Top view</text>
    <circle cx="${cx}" cy="${cy}" r="${78}" fill="none" stroke="var(--line)" stroke-dasharray="2 4"/>
    <ellipse cx="${cx}" cy="${cy}" rx="16" ry="9" fill="var(--bg3)" stroke="var(--fg2)" stroke-width="1.6"/>
    <path d="M${cx - 4} ${cy + 8}l4 7 4 -7" fill="var(--fg2)"/>
    <text x="${cx + 22}" y="${cy + 4}" class="cmp-cap">L</text><text x="${cx - 30}" y="${cy + 4}" class="cmp-cap">R</text>
    <text x="${cx}" y="${H - 6}" text-anchor="middle" class="cmp-cap">subject faces this way ↓</text>
    ${cam(pts[0], COL_A, 'A')}${cam(pts[1], COL_B, 'B')}
  </svg>`;
}

function table(a, b) {
  const ca = a.result.camera, cb = b.result.camera;
  const wrap = (d) => ((d + 540) % 360) - 180;
  const rows = [
    ['Tilt (+ down)', ca.pitch_deg.v, cb.pitch_deg.v, (v) => F.deg(v, 1), '°'],
    ['Roll', ca.roll_deg.v, cb.roll_deg.v, (v) => F.deg(v, 1), '°'],
    ['Height vs. eyes', ca.height_above_eye_m.v, cb.height_above_eye_m.v, (v) => F.sm(v), 'm'],
    ['Height above ground', ca.height_above_ground_m?.v, cb.height_above_ground_m?.v, (v) => F.m(v), 'm'],
    ['Distance to eyes', ca.distance_to_eyes_m.v, cb.distance_to_eyes_m.v, (v) => F.m(v), 'm'],
    ['Elevation', ca.elevation_deg.v, cb.elevation_deg.v, (v) => F.deg(v, 1), '°'],
    ['Side angle', ca.azimuth_deg.v, cb.azimuth_deg.v, (v) => F.deg(v, 0), '°a'],
    ['Lens (35 mm eq.)', a.intr.f35_mm, b.intr.f35_mm, (v) => `${Math.round(v)} mm`, 'mm'],
  ];
  const diff = (x, y, unit) => {
    if (x == null || y == null || !isFinite(x) || !isFinite(y)) return '—';
    let d = y - x;
    if (unit === '°a') d = wrap(d);
    const sign = d > 0 ? '+' : d < 0 ? '−' : '';
    if (unit === 'm') return `${sign}${Math.abs(d).toFixed(2)} m`;
    if (unit === 'mm') return `${sign}${Math.abs(d).toFixed(0)} mm`;
    return `${sign}${Math.abs(d).toFixed(1)}°`;
  };
  return `<table class="metrics cmp-table"><thead><tr><th></th><th style="color:${COL_A}">A</th><th style="color:${COL_B}">B</th><th>B − A</th></tr></thead><tbody>` +
    rows.map(([k, x, y, f, u]) => `<tr><td>${esc(k)}</td><td class="v">${x == null ? '—' : esc(f(x))}</td><td class="v">${y == null ? '—' : esc(f(y))}</td><td class="r">${diff(x, y, u)}</td></tr>`).join('') +
    `<tr><td>Angle</td><td class="v">${F.ANGLE[a.result.labels.angle]}</td><td class="v">${F.ANGLE[b.result.labels.angle]}</td><td></td></tr>` +
    `<tr><td>Framing</td><td class="v">${F.FRAMING[a.result.labels.framing]}</td><td class="v">${F.FRAMING[b.result.labels.framing]}</td><td></td></tr>` +
    '</tbody></table>';
}

/** deps: { decode(file, maxSide), drawOverlay(p, ctx, scale, on) } */
export async function openCompare(a, b, deps) {
  const dlg = document.getElementById('compare');
  const body = dlg.querySelector('.cmp-body');
  body.innerHTML = `
    <div class="cmp-photos">
      ${[a, b].map((p, i) => `<figure class="cmp-fig"><div class="cmp-canvas"><canvas data-i="${i}"></canvas></div>
        <figcaption><span class="cmp-badge" style="background:${i ? COL_B : COL_A}">${i ? 'B' : 'A'}</span> ${esc(p.name)}</figcaption></figure>`).join('')}
    </div>
    <p class="headline">${esc(summary(a, b))}</p>
    <div class="cmp-diagrams">${sideView(a, b)}${topView(a, b)}</div>
    ${table(a, b)}
    <p class="muted small">Each photo is measured relative to its own subject. If the subjects differ in height or the lens info is missing, compare the ranges in each photo’s detail view before trusting small differences.</p>`;
  if (!dlg.open) dlg.showModal();
  // draw both photos with their horizon / eye-line overlays
  await Promise.all([a, b].map(async (p, i) => {
    const cv = body.querySelector(`canvas[data-i="${i}"]`);
    try {
      const { bmp } = await deps.decode(p.file, 900);
      cv.width = bmp.width; cv.height = bmp.height;
      const ctx = cv.getContext('2d');
      ctx.drawImage(bmp, 0, 0);
      bmp.close();
      deps.drawOverlay(p, ctx, cv.width / p.det.w, (k) => k !== 'face');
    } catch { /* leave blank */ }
  }));
}
