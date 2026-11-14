// CamPose web app: batch queue, gallery cards, detail panel. 3D lives in viewer/.
import { initCore, analyze } from './core.js';
import { loadModels, detectPhoto, applyAssumptions, decode, delegateUsed } from './landmarks.js';
import * as F from './format.js';

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const IMG_RE = /\.(jpe?g|png|webp|heic|heif|avif|bmp|gif)$/i;

// ---------- settings ----------
const DEFAULTS = { model: 'lite', processor: 'auto', f35: 26, subjectH: null, mc: 64 };
let settings = { ...DEFAULTS };
try { Object.assign(settings, JSON.parse(localStorage.getItem('campose.settings') || '{}')); } catch { /* storage blocked */ }
const saveSettings = () => { try { localStorage.setItem('campose.settings', JSON.stringify(settings)); } catch { /* ignore */ } };

// ---------- state ----------
/** @type {Array<any>} */
const photos = [];
let nextId = 1;
let running = false;
let filter = 'all';
let sortKey = 'added';
let detailIdx = -1;
let coreReady = initCore();
let modelsReady = null;
const timings = [];

// ---------- adding files ----------
/** Recognise images by their first bytes, so files with no or wrong extensions still work. */
async function looksLikeImage(f) {
  if (f.type.startsWith('image/') || IMG_RE.test(f.name)) return true;
  if (f.size < 12 || f.size > 200e6) return false;
  const b = new Uint8Array(await f.slice(0, 16).arrayBuffer());
  const ascii = (o, n) => String.fromCharCode(...b.slice(o, o + n));
  return (b[0] === 0xff && b[1] === 0xd8) // JPEG
    || ascii(1, 3) === 'PNG'
    || (ascii(0, 4) === 'RIFF' && ascii(8, 4) === 'WEBP')
    || ascii(0, 3) === 'GIF' || ascii(0, 2) === 'BM'
    || (ascii(4, 4) === 'ftyp' && /hei|mif|avi/.test(ascii(8, 4))); // HEIC / AVIF
}

async function addFiles(files) {
  const all = [...files];
  const ok = await Promise.all(all.map(looksLikeImage));
  const list = all.filter((_, i) => ok[i]);
  if (!list.length) return;
  list.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name, undefined, { numeric: true }));
  for (const file of list) {
    const p = { id: nextId++, file, name: file.webkitRelativePath || file.name, status: 'queued', a: null, result: null };
    photos.push(p);
    $('#grid').appendChild(renderCard(p));
  }
  refreshChrome();
  pump();
}

async function readEntries(entry, out) {
  if (entry.isFile) {
    await new Promise((res) => entry.file((f) => {
      if (!/\.(txt|json|csv|md|db|ini|mp4|mov|pdf|zip)$/i.test(f.name) && !f.name.startsWith('.')) {
        // keep the folder path for display
        try { Object.defineProperty(f, 'webkitRelativePath', { value: entry.fullPath.replace(/^\//, '') }); } catch { /* ignore */ }
        out.push(f);
      }
      res();
    }, res));
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    for (;;) {
      const batch = await new Promise((res) => reader.readEntries(res, () => res([])));
      if (!batch.length) break;
      for (const e of batch) await readEntries(e, out);
    }
  }
}

// ---------- queue ----------
function setStatus(text, frac) {
  const s = $('#status');
  s.hidden = text == null;
  if (text != null) $('#statusText').textContent = text;
  if (frac != null) $('#statusBar').style.width = `${Math.round(frac * 100)}%`;
}

async function ensureModels() {
  if (!modelsReady) {
    modelsReady = (async () => {
      await coreReady;
      setStatus('Loading models… (first time only, then cached)', 0);
      await loadModels(settings, (p) => setStatus(`Downloading models… ${Math.round(p * 100)}%`, p));
    })().catch((e) => { modelsReady = null; throw e; });
  }
  return modelsReady;
}

async function pump() {
  if (running) return;
  running = true;
  try {
    await ensureModels();
    for (;;) {
      const p = photos.find((x) => x.status === 'queued');
      if (!p) break;
      p.status = 'running';
      updateCard(p);
      const done = photos.filter((x) => x.status === 'done' || x.status === 'error').length;
      const avg = timings.length ? timings.reduce((a, b) => a + b, 0) / timings.length : 0;
      setStatus(`Analyzing ${done + 1} of ${photos.length}${avg ? ` · ${Math.round(avg)} ms per photo (${delegateUsed})` : ''}`, done / photos.length);
      try {
        p.det = await detectPhoto(p.file, settings);
        timings.push(p.det.ms); if (timings.length > 20) timings.shift();
        p.a = p.a || { lensMode: 'auto', f35: settings.f35, subjectH: settings.subjectH, upright: true, mirrored: null, mc: settings.mc };
        solve(p);
      } catch (e) {
        console.error(p.name, e);
        p.status = 'error';
        p.error = /decode|source image/i.test(String(e)) ? 'Can’t decode this file (HEIC works in Safari only; convert to JPEG).' : 'Analysis failed.';
      }
      updateCard(p);
      refreshChrome();
      // let the page breathe between photos (keeps weak laptops responsive)
      await new Promise((r) => setTimeout(r, 0));
    }
    const ok = photos.filter((x) => x.status === 'done').length;
    setStatus(photos.length ? `Done: ${ok} of ${photos.length} photos analyzed` : null, 1);
    setTimeout(() => { if (!running) setStatus(null); }, 4000);
  } catch (e) {
    console.error(e);
    setStatus(`Couldn’t load the models: ${e.message || e}. Check your connection and reload.`, 0);
  } finally {
    running = false;
    refreshChrome();
  }
}

/** Geometry only (no inference): runs on every assumption change. */
function solve(p) {
  if (!p.det.hasPose && !p.det.hasFace) {
    p.status = 'error';
    p.error = 'No person found';
    return;
  }
  const { intr, mirrored } = applyAssumptions(p.det, p.a);
  p.intr = intr;
  p.mirrored = mirrored;
  const t = performance.now();
  const r = analyze(p.det.input);
  p.solveMs = performance.now() - t;
  if (!r.ok) { p.status = 'error'; p.error = r.error === 'no_person' ? 'No person found' : 'Couldn’t solve the camera'; return; }
  if (r.focal?.source === 'estimated') {
    // focal length came from the body fit: express it in the same units as EXIF/preset
    const k = r.focal.f_px / r.focal.f_input_px;
    const fr = p.det.w / p.det.W0; // focal values in the result are at analysis size
    const fW = r.focal.f_px / fr;
    p.intr = {
      ...intr, source: 'estimated', f_px: fW, f35_mm: intr.f35_mm * k,
      f35_lo: intr.f35_mm * r.focal.lo_px / r.focal.f_input_px, f35_hi: intr.f35_mm * r.focal.hi_px / r.focal.f_input_px,
      hfov_deg: 2 * Math.atan(p.det.W0 / 2 / fW) * 180 / Math.PI, vfov_deg: 2 * Math.atan(p.det.H0 / 2 / fW) * 180 / Math.PI,
    };
  }
  p.result = r;
  p.status = 'done';
}

// ---------- cards ----------
function renderCard(p) {
  const c = el('article', 'card');
  c.tabIndex = 0;
  c.dataset.id = p.id;
  c.setAttribute('role', 'button');
  c.addEventListener('click', () => openDetail(p));
  c.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDetail(p); } });
  p.card = c;
  updateCard(p);
  return c;
}

function sideGlyph(r) {
  // side view: subject on the right, camera placed by horizontal distance and height vs. eyes
  const c = r.camera;
  const dh = c.horizontal_distance_m.v, he = c.height_above_eye_m.v;
  const W = 200, H = 44, gy = 40, px = 172, eyeY = 16;
  const sc = Math.min(140 / Math.max(dh, 0.3), 14 / Math.max(Math.abs(he) / (dh || 1) * 10, 0.01) * 10, 60);
  let cx = px - dh * sc, cy = eyeY - he * sc;
  cx = Math.max(10, cx); cy = Math.max(4, Math.min(gy - 2, cy));
  const pitch = c.pitch_deg.v * Math.PI / 180;
  const fx = Math.cos(pitch) * 16, fy = Math.sin(pitch) * 16;
  return `<svg class="side-glyph" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
    <line x1="4" y1="${gy}" x2="${W - 4}" y2="${gy}" stroke="var(--line)" stroke-width="1.5"/>
    <line x1="${cx}" y1="${cy}" x2="${px}" y2="${eyeY}" stroke="var(--warm)" stroke-width="1.2" stroke-dasharray="3 2.5"/>
    <line x1="${px - 14}" y1="${eyeY}" x2="${px + 10}" y2="${eyeY}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="1.5 2"/>
    <circle cx="${px}" cy="10" r="4" fill="none" stroke="var(--fg2)" stroke-width="1.6"/>
    <path d="M${px} 14v14M${px} 28l-4 12M${px} 28l4 12M${px - 6} 19h12" stroke="var(--fg2)" stroke-width="1.6" fill="none" stroke-linecap="round"/>
    <line x1="${cx}" y1="${cy}" x2="${cx + fx}" y2="${cy + fy}" stroke="var(--accent)" stroke-width="1.4"/>
    <rect x="${cx - 6}" y="${cy - 4}" width="12" height="8" rx="2" fill="var(--accent)"/>
  </svg>`;
}

function thumbOverlay(p) {
  const r = p.result, d = p.det;
  if (!r) return '';
  const { w, h } = d;
  const line = (l, color, dash) => {
    const [a, b, c] = l;
    if (Math.abs(b) < 1e-9) return '';
    const y0 = -(c) / b, y1 = -(a * w + c) / b;
    return `<line x1="0" y1="${y0}" x2="${w}" y2="${y1}" stroke="${color}" stroke-width="${w / 160}" ${dash ? `stroke-dasharray="${w / 50} ${w / 70}"` : ''}/>`;
  };
  const [ex, ey] = r.overlay.eye_px;
  return `<svg class="ov" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid slice">
    ${line(r.overlay.horizon, '#4ea3ff', false)}${line(r.overlay.eye_line, '#ffb35c', true)}
    <circle cx="${ex}" cy="${ey}" r="${w / 90}" fill="#ffb35c"/></svg>`;
}

function updateCard(p) {
  const c = p.card;
  if (!c) return;
  c.className = `card state-${p.status}`;
  const img = p.det?.thumb ? `<img src="${p.det.thumb.url}" alt="" loading="lazy" decoding="async">` : '';
  const fname = `<div class="fname" title="${esc(p.name)}">${esc(p.name)}</div>`;
  if (p.status !== 'done') {
    const msg = p.status === 'error' ? esc(p.error) : p.status === 'running' ? 'Analyzing…' : 'Waiting…';
    c.innerHTML = `<div class="thumb">${img}</div><div class="body">${fname}<div class="pending">${msg}</div></div>`;
    return;
  }
  const r = p.result, cam = r.camera, l = r.labels;
  const lensSrc = F.LENS_SOURCE[p.intr.source] || p.intr.source;
  const warn = r.warnings.length ? `<span class="tag warn" title="${esc(r.warnings.map((w) => F.WARN[w] || w).join('\n'))}">${r.warnings.length} note${r.warnings.length > 1 ? 's' : ''}</span>` : '';
  c.innerHTML = `
    <div class="thumb">${img}${thumbOverlay(p)}
      <span class="badge">${F.ANGLE[l.angle]}</span>${l.dutch ? `<span class="badge dutch">Dutch ${Math.abs(cam.roll_deg.v).toFixed(0)}°</span>` : ''}
    </div>
    <div class="body">
      ${fname}
      <div class="tags"><span class="tag">${F.FRAMING[l.framing]}</span><span class="tag">${F.SIDE_SHORT[l.side]}</span>${warn}</div>
      ${sideGlyph(r)}
      <dl class="stats">
        <div><dt>Tilt</dt><dd>${F.tilt(cam.pitch_deg.v)}</dd></div>
        <div><dt>vs. eye level</dt><dd>${F.sm(cam.height_above_eye_m.v)}</dd></div>
        <div><dt>Distance</dt><dd>${F.m(cam.distance_to_eyes_m.v, 1)}</dd></div>
        <div><dt>Lens</dt><dd title="${esc(lensSrc)}">${p.intr.source === 'estimated' ? '≈' : ''}${Math.round(p.intr.f35_mm)} mm <span class="muted small">${p.intr.source === 'preset' ? '?' : ''}</span></dd></div>
      </dl>
    </div>`;
  c.setAttribute('aria-label', `${p.name}: ${F.headline(r)}`);
}

// ---------- filters / sort ----------
const FILTERS = [
  ['all', 'All'], ['eye_level', 'Eye level'], ['high', 'High'], ['low', 'Low'],
  ['overhead', 'Overhead'], ['worms_eye', "Worm's-eye"], ['dutch', 'Dutch'], ['issues', 'Needs a look'],
];
function matches(p, f) {
  if (f === 'all') return true;
  if (p.status !== 'done') return f === 'issues' && p.status === 'error';
  const l = p.result.labels;
  if (f === 'high') return l.angle === 'high' || l.angle === 'steep_high';
  if (f === 'low') return l.angle === 'low' || l.angle === 'steep_low';
  if (f === 'dutch') return l.dutch;
  if (f === 'issues') return p.result.warnings.length > 0;
  return l.angle === f;
}
function renderFilters() {
  const box = $('#filters');
  box.innerHTML = '';
  for (const [k, label] of FILTERS) {
    const n = photos.filter((p) => matches(p, k)).length;
    if (k !== 'all' && k !== filter && n === 0) continue;
    const b = el('button', `chip${filter === k ? ' on' : ''}`, `${label} <span class="n">${n}</span>`);
    b.type = 'button';
    b.setAttribute('aria-pressed', filter === k);
    b.onclick = () => { filter = k; applyView(); };
    box.appendChild(b);
  }
}
function sortVal(p) {
  const c = p.result?.camera;
  switch (sortKey) {
    case 'name': return p.name;
    case 'dist': return c ? c.distance_to_eyes_m.v : Infinity;
    case 'height': return c ? -c.height_above_eye_m.v : Infinity;
    case 'pitch': return c ? -c.pitch_deg.v : Infinity;
    case 'azim': return c ? c.azimuth_deg.v : Infinity;
    default: return p.id;
  }
}
function visiblePhotos() {
  const list = photos.filter((p) => matches(p, filter));
  if (sortKey !== 'added') {
    list.sort((a, b) => {
      const x = sortVal(a), y = sortVal(b);
      return typeof x === 'string' ? x.localeCompare(y, undefined, { numeric: true }) : x - y;
    });
  }
  return list;
}
function applyView() {
  const vis = visiblePhotos();
  const set = new Set(vis);
  const grid = $('#grid');
  for (const p of photos) p.card.hidden = !set.has(p);
  if (sortKey !== 'added' || grid.dataset.sorted) {
    const frag = document.createDocumentFragment();
    for (const p of vis) frag.appendChild(p.card);
    for (const p of photos) if (!set.has(p)) frag.appendChild(p.card);
    grid.appendChild(frag);
    grid.dataset.sorted = sortKey !== 'added' ? '1' : '';
  }
  renderFilters();
  $('#count').textContent = `${vis.length} photo${vis.length === 1 ? '' : 's'}`;
}
let chromeTimer = 0;
function refreshChrome() {
  const has = photos.length > 0;
  $('#empty').hidden = has;
  $('#toolbar').hidden = !has;
  $('#btnExport').disabled = !photos.some((p) => p.status === 'done');
  $('#btnClear').disabled = !has || running;
  clearTimeout(chromeTimer);
  chromeTimer = setTimeout(applyView, 60); // batch updates during big folder runs
}

// ---------- detail ----------
const dlg = $('#detail');
let detailImg = null; // { bmp, scale }

async function openDetail(p) {
  if (p.status === 'queued' || p.status === 'running') return;
  const vis = visiblePhotos();
  detailIdx = vis.indexOf(p);
  dlg.dataset.id = p.id;
  if (!dlg.open) dlg.showModal();
  $('#dTitle').textContent = p.name;
  detailImg?.bmp?.close?.();
  detailImg = null;
  renderDetail(p);
  try {
    const { bmp } = await decode(p.file, 1600);
    if (dlg.dataset.id != p.id) { bmp.close(); return; }
    detailImg = { bmp };
    drawDetailCanvas(p);
  } catch { /* keep thumbnail-less view */ }
}

function curDetail() { return photos.find((x) => x.id == dlg.dataset.id); }

function drawDetailCanvas(p) {
  const cv = $('#dCanvas');
  const ctx = cv.getContext('2d');
  const d = p.det;
  const bmp = detailImg?.bmp;
  const W = bmp ? bmp.width : d.thumb.w, H = bmp ? bmp.height : d.thumb.h;
  cv.width = W; cv.height = H;
  if (bmp) ctx.drawImage(bmp, 0, 0);
  else { const im = new Image(); im.onload = () => { if (!detailImg) { ctx.drawImage(im, 0, 0, W, H); drawOverlay(p, ctx, W / d.w); } }; im.src = d.thumb.url; return; }
  drawOverlay(p, ctx, W / d.w);
}

const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [27, 29], [29, 31], [27, 31], [24, 26], [26, 28], [28, 30], [30, 32], [28, 32], [0, 2], [0, 5], [2, 7], [5, 8], [9, 10]];
function sideColor(i, j) {
  const L = (k) => k % 2 === 1 && k >= 11; // odd indices ≥ 11 are the subject's left
  if (L(i) && L(j)) return '#ffb35c';
  const R = (k) => k % 2 === 0 && k >= 12;
  if (R(i) && R(j)) return '#4ea3ff';
  return '#e8ebf1';
}

function drawOverlay(p, ctx, s) {
  const r = p.result;
  if (!r) return;
  const on = (k) => $(`#ovToggles input[data-ov="${k}"]`).checked;
  const W = ctx.canvas.width;
  const lw = Math.max(1.5, W / 500);
  const drawLine = (l, color, dash, label) => {
    const [a, b, c] = l;
    if (Math.abs(b) < 1e-9) return;
    const y = (x) => -(a * x + c) / b;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = lw * 1.2; ctx.setLineDash(dash ? [lw * 6, lw * 4] : []);
    ctx.beginPath(); ctx.moveTo(0, y(0) * s); ctx.lineTo(W, y(W / s) * s); ctx.stroke();
    ctx.font = `600 ${Math.round(lw * 7)}px system-ui`; ctx.fillStyle = color;
    ctx.shadowColor = 'rgba(0,0,0,.7)'; ctx.shadowBlur = 4;
    ctx.fillText(label, lw * 6, y(lw * 6 / s) * s - lw * 3);
    ctx.restore();
  };
  if (on('horizon')) drawLine(r.overlay.horizon, '#4ea3ff', false, 'Horizon (camera height)');
  if (on('eye')) drawLine(r.overlay.eye_line, '#ffb35c', true, 'Subject eye level');
  const inp = p.det.input;
  if (on('face') && p.det.hasFace) {
    ctx.fillStyle = 'rgba(120, 230, 200, .75)';
    const rr = Math.max(1, lw * 0.6);
    for (let i = 0; i < 478; i += 2) {
      const u = inp[214 + i * 3] * s, v = inp[214 + i * 3 + 1] * s;
      ctx.fillRect(u - rr / 2, v - rr / 2, rr, rr);
    }
  }
  if (on('skeleton') && r.overlay.reproj_px) {
    const q = r.overlay.reproj_px;
    ctx.lineWidth = lw * 1.6; ctx.lineCap = 'round';
    for (const [i, j] of BONES) {
      ctx.strokeStyle = sideColor(i, j);
      ctx.beginPath(); ctx.moveTo(q[i * 2] * s, q[i * 2 + 1] * s); ctx.lineTo(q[j * 2] * s, q[j * 2 + 1] * s); ctx.stroke();
    }
    // detected points (white rings) vs. fitted model (filled): their gap is the fit error
    for (let i = 0; i < 33; i++) {
      const vis = inp[16 + i * 3 + 2];
      if (vis < 0.3) continue;
      ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.lineWidth = lw * 0.8;
      ctx.beginPath(); ctx.arc(inp[16 + i * 3] * s, inp[16 + i * 3 + 1] * s, lw * 2.2, 0, 7); ctx.stroke();
    }
  }
  const [ex, ey] = r.overlay.eye_px;
  ctx.fillStyle = '#ffb35c';
  ctx.beginPath(); ctx.arc(ex * s, ey * s, lw * 2.5, 0, 7); ctx.fill();
}

function renderDetail(p) {
  const r = p.result;
  $('#dPrev').disabled = detailIdx <= 0;
  $('#dNext').disabled = detailIdx < 0 || detailIdx >= visiblePhotos().length - 1;
  $('#btn3d').disabled = !r;
  $('#btnCopy').disabled = !r;
  drawDetailCanvas(p);
  if (!r) {
    $('#dHeadline').textContent = p.error || 'No result';
    $('#dTable').innerHTML = '';
    $('#dWarnings').innerHTML = '';
    $('#dMeta').textContent = '';
    return;
  }
  $('#dHeadline').textContent = F.headline(r);
  const c = r.camera;
  const rows = [
    ['Tilt (pitch)', F.tilt(c.pitch_deg.v), F.range(c.pitch_deg, (v) => F.deg(v, 1)), 'from posture and face orientation (' + r.gravity.sources.join(', ').replace('assumed_level', 'assumed level') + ')'],
    ['Roll', F.deg(c.roll_deg.v, 1), F.range(c.roll_deg, (v) => F.deg(v, 1)), r.labels.dutch ? 'Dutch angle' : ''],
    ['Height vs. subject’s eyes', F.sm(c.height_above_eye_m.v), F.range(c.height_above_eye_m, (v) => F.sm(v)), ''],
    ['Height above ground', c.height_above_ground_m ? F.m(c.height_above_ground_m.v) : '—', c.height_above_ground_m ? F.range(c.height_above_ground_m, (v) => F.m(v)) : 'feet not visible', ''],
    ['Distance to eyes', F.m(c.distance_to_eyes_m.v), F.range(c.distance_to_eyes_m, (v) => F.m(v)), r.fit.iris_used ? 'iris size + body' : r.mode === 'face' ? 'eye spacing' : 'body size'],
    ['Horizontal distance', F.m(c.horizontal_distance_m.v), F.range(c.horizontal_distance_m, (v) => F.m(v)), ''],
    ['Elevation angle', F.deg(c.elevation_deg.v, 1), F.range(c.elevation_deg, (v) => F.deg(v, 1)), F.ANGLE[r.labels.angle]],
    ['Side angle (azimuth)', F.deg(c.azimuth_deg.v, 0), F.range(c.azimuth_deg, (v) => F.deg(v, 0)), F.SIDE[r.labels.side] + (p.mirrored ? ' (mirrored)' : '')],
    ['Lens', `${Math.round(p.intr.f35_mm)} mm eq.`, p.intr.source === 'estimated' ? `${Math.round(p.intr.f35_lo)} … ${Math.round(p.intr.f35_hi)} mm` : `${p.intr.hfov_deg.toFixed(0)}° × ${p.intr.vfov_deg.toFixed(0)}° FOV`, (F.LENS_SOURCE[p.intr.source] || p.intr.source) + (p.intr.device ? ` · ${p.intr.device}` : '')],
    ['Subject height', F.m(r.subject.height_m), '', { given: 'entered by you', assumed: 'assumed (edit below)', 'body+iris': 'implied by body + iris size' }[r.subject.height_source] || r.subject.height_source],
  ];
  $('#dTable').innerHTML = '<tbody>' + rows.map(([k, v, rg, src]) =>
    `<tr><td>${esc(k)}${src ? `<span class="src">${esc(src)}</span>` : ''}</td><td class="v">${esc(v)}</td><td class="r">${esc(rg)}</td></tr>`).join('') + '</tbody>';
  $('#dWarnings').innerHTML = r.warnings.map((w) => `<div class="warnbox">${esc(F.WARN[w] || w)}</div>`).join('');
  // assumptions
  const a = p.a;
  const lensSel = $('#aLens');
  const autoLabel = a.lensMode === 'preset' ? 'Auto (EXIF or estimate)'
    : p.intr.source === 'estimated' ? `Auto: estimated ≈${Math.round(p.intr.f35_mm)} mm`
      : p.intr.source === 'preset' ? `Auto: default ${settings.f35} mm (no EXIF)` : `Auto: EXIF ${Math.round(p.intr.f35_mm)} mm`;
  lensSel.innerHTML = `<option value="auto">${autoLabel}</option>` +
    F.LENS_PRESETS.map(([n, f]) => `<option value="${f}">${n} · ${f} mm</option>`).join('');
  lensSel.value = a.lensMode === 'preset' ? String(a.f35) : 'auto';
  if (lensSel.value === '' && a.lensMode === 'preset') {
    lensSel.insertAdjacentHTML('beforeend', `<option value="${a.f35}">Custom · ${a.f35} mm</option>`);
    lensSel.value = String(a.f35);
  }
  $('#aHeight').value = a.subjectH ?? '';
  $('#aUpright').checked = a.upright;
  $('#aMirror').checked = !!p.mirrored;
  $('#dMeta').textContent = `Mode: ${r.mode} · fit error ${r.fit.reproj_rmse_px != null ? r.fit.reproj_rmse_px.toFixed(1) + ' px' : 'n/a'} · ` +
    `${r.fit.landmarks_used} body landmarks · ${p.det.people > 1 ? `${p.det.people} people found, using the largest · ` : ''}` +
    `analysis ${p.det.w}×${p.det.h} px · detect ${Math.round(p.det.ms)} ms · solve ${p.solveMs.toFixed(1)} ms`;
}

function onAssumption() {
  const p = curDetail();
  if (!p || !p.det) return;
  const lv = $('#aLens').value;
  p.a.lensMode = lv === 'auto' ? 'auto' : 'preset';
  if (lv !== 'auto') p.a.f35 = +lv;
  else p.a.f35 = settings.f35;
  const h = parseFloat($('#aHeight').value);
  p.a.subjectH = h > 0.5 && h < 2.6 ? h : null;
  p.a.upright = $('#aUpright').checked;
  p.a.mirrored = $('#aMirror').checked;
  solve(p);
  renderDetail(p);
  updateCard(p);
  refreshChrome();
}

// ---------- export ----------
function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function exportAll(kind) {
  const done = photos.filter((p) => p.status === 'done');
  if (kind === 'json') {
    const out = done.map((p) => {
      const { scene, overlay, ...rest } = p.result;
      return { file: p.name, intrinsics: p.intr, ...rest, pose: { R_wc: scene.R_wc, cam_pos_w: scene.cam_pos_w } };
    });
    download('campose-results.json', JSON.stringify(out, null, 2), 'application/json');
  } else {
    const keys = ['pitch_deg', 'roll_deg', 'height_above_eye_m', 'height_above_ground_m', 'distance_to_eyes_m', 'horizontal_distance_m', 'elevation_deg', 'azimuth_deg'];
    const head = ['file', 'mode', ...keys.flatMap((k) => [k, `${k}_p10`, `${k}_p90`]), 'angle', 'dutch', 'side', 'framing', 'f35_mm', 'lens_source', 'subject_height_m', 'warnings'];
    const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
    const rows = done.map((p) => {
      const r = p.result, c = r.camera;
      return [q(p.name), r.mode, ...keys.flatMap((k) => c[k] ? [c[k].v, c[k].lo, c[k].hi] : ['', '', '']),
        r.labels.angle, r.labels.dutch, r.labels.side, r.labels.framing, p.intr.f35_mm.toFixed(1), p.intr.source,
        r.subject.height_m, q(r.warnings.join(' '))].join(',');
    });
    download('campose-results.csv', [head.join(','), ...rows].join('\n'), 'text/csv');
  }
}

// ---------- wiring ----------
$('#btnFiles').onclick = () => $('#inFiles').click();
$('#btnFolder').onclick = () => $('#inFolder').click();
document.querySelectorAll('[data-pick]').forEach((b) => { b.onclick = () => $(b.dataset.pick === 'folder' ? '#inFolder' : '#inFiles').click(); });
$('#inFiles').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
$('#inFolder').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
$('#sort').onchange = (e) => { sortKey = e.target.value; applyView(); };
$('#btnClear').onclick = () => {
  if (running) return;
  for (const p of photos) { if (p.det?.thumb) URL.revokeObjectURL(p.det.thumb.url); p.card.remove(); }
  photos.length = 0; filter = 'all';
  refreshChrome();
};
$('#btnExport').onclick = (e) => { e.stopPropagation(); $('#exportMenu').hidden = !$('#exportMenu').hidden; };
document.addEventListener('click', () => { $('#exportMenu').hidden = true; });
$('#exportMenu').onclick = (e) => { const k = e.target.dataset.export; if (k) exportAll(k); };

$('#dClose').onclick = () => dlg.close();
dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
dlg.addEventListener('close', () => { detailImg?.bmp?.close?.(); detailImg = null; });
const step = (d) => { const vis = visiblePhotos(); const n = vis[detailIdx + d]; if (n) openDetail(n); };
$('#dPrev').onclick = () => step(-1);
$('#dNext').onclick = () => step(1);
dlg.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select')) return;
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
});
$('#ovToggles').onchange = () => { const p = curDetail(); if (p) drawDetailCanvas(p); };
['#aLens', '#aHeight', '#aUpright', '#aMirror'].forEach((s) => $(s).addEventListener('change', onAssumption));
$('#btnCopy').onclick = async () => {
  const p = curDetail();
  if (!p?.result) return;
  const { overlay, ...rest } = p.result;
  try { await navigator.clipboard.writeText(JSON.stringify({ file: p.name, intrinsics: p.intr, ...rest }, null, 2)); $('#btnCopy').textContent = 'Copied'; }
  catch { $('#btnCopy').textContent = 'Copy failed'; }
  setTimeout(() => { $('#btnCopy').textContent = 'Copy JSON'; }, 1500);
};
$('#btn3d').onclick = async () => {
  const p = curDetail();
  if (!p?.result) return;
  const btn = $('#btn3d');
  btn.disabled = true;
  const label = btn.innerHTML;
  btn.innerHTML = 'Building…';
  try {
    const { openViewer } = await import('./viewer/viewer.js');
    const { preparePhotoFor3D } = await import('./landmarks.js');
    const assets = await preparePhotoFor3D(p.file);
    dlg.close();
    openViewer(p, assets, () => openDetail(p));
  } catch (e) {
    console.error(e);
    alert('Couldn’t build the 3D scene: ' + (e.message || e));
  } finally {
    btn.innerHTML = label;
    btn.disabled = false;
  }
};

// settings dialog
const lensOpts = F.LENS_PRESETS.map(([n, f]) => `<option value="${f}">${n} · ${f} mm</option>`).join('');
$('#sLens').innerHTML = lensOpts;
$('#btnSettings').onclick = () => {
  $('#sModel').value = settings.model; $('#sProc').value = settings.processor;
  $('#sLens').value = String(settings.f35); $('#sHeight').value = settings.subjectH ?? ''; $('#sMc').value = String(settings.mc);
  $('#settings').showModal();
};
$('#settings').addEventListener('close', () => {
  const prev = { ...settings };
  settings.model = $('#sModel').value; settings.processor = $('#sProc').value;
  settings.f35 = +$('#sLens').value; settings.mc = +$('#sMc').value;
  const h = parseFloat($('#sHeight').value); settings.subjectH = h > 0.5 && h < 2.6 ? h : null;
  saveSettings();
  if (prev.model !== settings.model || prev.processor !== settings.processor) {
    modelsReady = null;
    // re-run inference on everything with the new model
    for (const p of photos) if (p.status !== 'queued') { p.status = 'queued'; updateCard(p); }
    pump();
    return;
  }
  // geometry-only defaults: re-solve photos without their own override
  for (const p of photos) {
    if (!p.det || !p.a) continue;
    if (p.a.lensMode === 'auto') p.a.f35 = settings.f35;
    if (prev.subjectH === p.a.subjectH) p.a.subjectH = settings.subjectH;
    p.a.mc = settings.mc;
    if (p.status === 'done' || p.result) { solve(p); updateCard(p); }
  }
  refreshChrome();
});

// drag & drop (files and folders)
let dragDepth = 0;
window.addEventListener('dragenter', (e) => { if ([...e.dataTransfer.types].includes('Files')) { dragDepth++; $('#dropOverlay').hidden = false; } });
window.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $('#dropOverlay').hidden = true; });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0; $('#dropOverlay').hidden = true;
  const items = [...(e.dataTransfer.items || [])];
  const entries = items.map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  if (entries.length) {
    const out = [];
    for (const en of entries) await readEntries(en, out);
    addFiles(out);
  } else {
    addFiles(e.dataTransfer.files);
  }
});

// Dev/test hook: ?load=url1,url2 fetches same-origin images into the queue.
const loadParam = new URLSearchParams(location.search).get('load');
if (loadParam) {
  (async () => {
    const files = [];
    for (const u of loadParam.split(',')) {
      const url = new URL(u, location.href);
      if (url.origin !== location.origin) continue;
      const b = await (await fetch(url)).blob();
      files.push(new File([b], url.pathname.split('/').pop(), { type: b.type }));
    }
    addFiles(files);
  })();
}

refreshChrome();
// warm up the models in the background so the first photo is quick
coreReady.then(() => { if (!photos.length) ensureModels().then(() => setStatus(null)).catch(() => {}); });
// for debugging in DevTools
window.__campose = { photos, settings };
