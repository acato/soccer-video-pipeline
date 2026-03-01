"""
Serves the embedded monitoring UI at GET /ui
A single-page HTML/JS app that polls the jobs API.
No build step required — served directly from this string.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Soccer Pipeline Monitor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e1e4e8; min-height: 100vh; }
  header { background: #161b22; border-bottom: 1px solid #30363d;
           padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 20px; font-weight: 600; }
  header .badge { background: #238636; color: white; font-size: 11px;
                  padding: 2px 8px; border-radius: 12px; font-weight: 600; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
  .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
               padding: 16px; text-align: center; }
  .stat-card .value { font-size: 32px; font-weight: 700; line-height: 1; }
  .stat-card .label { font-size: 12px; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-card.pending .value { color: #d29922; }
  .stat-card.processing .value { color: #388bfd; }
  .stat-card.complete .value { color: #3fb950; }
  .stat-card.failed .value { color: #f85149; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
  .panel-header { padding: 12px 16px; border-bottom: 1px solid #30363d;
                  display: flex; justify-content: space-between; align-items: center; }
  .panel-header h2 { font-size: 15px; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 10px 16px; font-size: 12px; color: #8b949e;
       text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #21262d; }
  td { padding: 12px 16px; border-bottom: 1px solid #21262d; font-size: 14px; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }
  .status { display: inline-flex; align-items: center; gap: 6px; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-pending { background: #d29922; }
  .dot-ingesting, .dot-detecting, .dot-segmenting, .dot-assembling { background: #388bfd; animation: pulse 1.5s infinite; }
  .dot-complete { background: #3fb950; }
  .dot-failed { background: #f85149; }
  .dot-paused { background: #d29922; }
  .dot-cancelled { background: #8b949e; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .progress-bar { width: 80px; height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; }
  .progress-fill { height: 100%; background: #388bfd; border-radius: 3px; transition: width 0.5s; }
  .btn { padding: 5px 12px; border-radius: 6px; border: 1px solid #30363d; cursor: pointer;
         font-size: 12px; background: #21262d; color: #e1e4e8; transition: background 0.15s; }
  .btn:hover { background: #30363d; }
  .btn-sm { padding: 3px 8px; font-size: 11px; }
  .btn-warn { border-color: #d29922; color: #d29922; }
  .btn-warn:hover { background: #d2992222; }
  .btn-danger { border-color: #f85149; color: #f85149; }
  .btn-danger:hover { background: #f8514922; }
  .reel-links { display: flex; gap: 6px; flex-wrap: wrap; }
  .reel-link { color: #388bfd; text-decoration: none; font-size: 12px;
               padding: 2px 6px; border: 1px solid #388bfd33; border-radius: 4px; }
  .reel-link:hover { background: #388bfd22; }
  .empty-state { padding: 48px 16px; text-align: center; color: #8b949e; }
  .submit-form { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 20px; margin-bottom: 24px; }
  .submit-form h2 { font-size: 15px; font-weight: 600; margin-bottom: 16px; }
  .form-row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
  .form-group { flex: 1; min-width: 0; }
  label { display: block; font-size: 12px; color: #8b949e; margin-bottom: 6px; }
  input[type=text], input[type=number] { width: 100%; padding: 8px 12px; background: #0d1117; border: 1px solid #30363d;
                     border-radius: 6px; color: #e1e4e8; font-size: 14px; }
  input[type=text]:focus, input[type=number]:focus { outline: none; border-color: #388bfd; }
  select { width: 100%; padding: 8px 12px; background: #0d1117; border: 1px solid #30363d;
           border-radius: 6px; color: #e1e4e8; font-size: 14px; }
  select:focus { outline: none; border-color: #388bfd; }
  .checkbox-group { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .checkbox-group label { display: flex; align-items: center; gap: 4px; margin-bottom: 0;
                          font-size: 12px; color: #e1e4e8; cursor: pointer; }
  .checkbox-group input[type=checkbox] { accent-color: #388bfd; width: 14px; height: 14px; cursor: pointer; }
  .btn-primary { background: #238636; border-color: #238636; color: white; padding: 8px 16px; }
  .btn-primary:hover { background: #2ea043; }
  .btn-outline { background: transparent; border: 1px solid #388bfd; color: #388bfd; padding: 4px 10px; font-size: 12px; }
  .btn-outline:hover { background: #388bfd22; }
  .btn-outline.active { background: #388bfd33; }
  .toast { position: fixed; bottom: 24px; right: 24px; background: #238636; color: white;
           padding: 12px 16px; border-radius: 8px; font-size: 14px; opacity: 0; transition: opacity 0.3s;
           max-width: 320px; z-index: 100; }
  .toast.show { opacity: 1; }
  .toast.error { background: #da3633; }
  #refresh-indicator { font-size: 12px; color: #8b949e; }
  .section-label { font-size: 13px; font-weight: 600; margin-top: 16px; margin-bottom: 8px; }
  .jersey-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
  .jersey-col { display: flex; flex-direction: column; gap: 8px; }
  .jersey-col h3 { font-size: 13px; font-weight: 600; color: #8b949e; }
  .reel-cards { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  .reel-card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px;
               display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .reel-card .reel-name { font-weight: 600; }
  .reel-card .reel-types { font-size: 11px; color: #8b949e; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .reel-card .remove-reel { cursor: pointer; color: #f85149; font-size: 14px; margin-left: 4px; }
  .preset-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
  .event-badge { display: inline-block; font-size: 10px; padding: 1px 5px; border-radius: 3px;
                 background: #21262d; color: #8b949e; border: 1px solid #30363d; }
  .event-badge.gk { background: #238636; color: white; border-color: #238636; }
  .event-badge.hl { background: #388bfd33; color: #388bfd; border-color: #388bfd55; }
</style>
</head>
<body>
<header>
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="#3fb950" stroke-width="2"/>
    <path d="M12 2C12 2 8 6 8 12s4 10 4 10" stroke="#3fb950" stroke-width="1.5"/>
    <path d="M2 12h20M7 4.5l10 15M17 4.5L7 19.5" stroke="#3fb950" stroke-width="1" opacity="0.5"/>
  </svg>
  <h1>Soccer Pipeline Monitor</h1>
  <span class="badge">LIVE</span>
  <span id="refresh-indicator" style="margin-left:auto">Refreshing...</span>
</header>

<div class="container">
  <div class="submit-form" id="submit-form-area">
    <h2>Submit Match</h2>

    <!-- Row 1: video + game start -->
    <div class="form-row">
      <div class="form-group" style="flex:3">
        <label>Video File</label>
        <select id="nas-path">
          <option value="">Loading files...</option>
        </select>
      </div>
      <div class="form-group" style="flex:0 0 120px">
        <label>Game start (min)</label>
        <input type="number" id="game-start-min" value="0" min="0" step="0.5" />
      </div>
    </div>

    <!-- Row 2: Jersey config -->
    <div class="section-label">Jersey Colors</div>
    <div class="jersey-grid">
      <div class="jersey-col">
        <h3>Your Team</h3>
        <div class="form-row">
          <div class="form-group">
            <label>Team Name</label>
            <input type="text" id="team-name" value="My Team" />
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Outfield</label>
            <select id="team-outfield"></select>
          </div>
          <div class="form-group">
            <label>GK</label>
            <select id="team-gk"></select>
          </div>
        </div>
      </div>
      <div class="jersey-col">
        <h3>Opponent</h3>
        <div class="form-row">
          <div class="form-group">
            <label>Team Name</label>
            <input type="text" id="opp-name" value="Opponent" />
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Outfield</label>
            <select id="opp-outfield"></select>
          </div>
          <div class="form-group">
            <label>GK</label>
            <select id="opp-gk"></select>
          </div>
        </div>
      </div>
    </div>

    <!-- Row 3: Reel builder -->
    <div class="section-label">Reels</div>
    <div class="preset-row">
      <button class="btn btn-outline" onclick="addPresetReel('keeper')">+ All GK</button>
      <button class="btn btn-outline" onclick="addPresetReel('highlights')">+ Highlights</button>
      <button class="btn btn-outline" onclick="addPresetReel('saves')">+ Saves Only</button>
      <button class="btn btn-outline" onclick="addPresetReel('distribution')">+ Distribution</button>
      <button class="btn btn-outline" onclick="addCustomReel()">+ Custom Reel...</button>
    </div>
    <div class="reel-cards" id="reel-cards"></div>

    <!-- Submit -->
    <div style="margin-top:16px; display:flex; justify-content:flex-end">
      <button class="btn btn-primary" onclick="submitJob()">Submit Job</button>
    </div>
  </div>

  <div class="stats-row">
    <div class="stat-card pending"><div class="value" id="stat-pending">-</div><div class="label">Pending</div></div>
    <div class="stat-card processing"><div class="value" id="stat-processing">-</div><div class="label">Processing</div></div>
    <div class="stat-card complete"><div class="value" id="stat-complete">-</div><div class="label">Complete</div></div>
    <div class="stat-card failed"><div class="value" id="stat-failed">-</div><div class="label">Failed</div></div>
  </div>

  <div class="panel">
    <div class="panel-header">
      <h2>Jobs</h2>
      <button class="btn btn-sm" onclick="loadJobs()">Refresh</button>
    </div>
    <table>
      <thead>
        <tr>
          <th>File</th>
          <th>Status</th>
          <th>Progress</th>
          <th>Reels</th>
          <th>Created</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody id="jobs-table-body">
        <tr><td colspan="6" class="empty-state">Loading...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="toast" id="toast"></div>

<!-- Custom reel dialog -->
<div id="custom-reel-dialog" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:50; display:none; align-items:center; justify-content:center;">
  <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; width:400px; max-height:80vh; overflow-y:auto;">
    <h3 style="font-size:15px; margin-bottom:12px;">Custom Reel</h3>
    <div class="form-group" style="margin-bottom:12px;">
      <label>Reel Name</label>
      <input type="text" id="custom-reel-name" placeholder="e.g. deflections" />
    </div>
    <div class="section-label">Goalkeeper Events</div>
    <div class="checkbox-group" id="custom-gk-events" style="margin-bottom:12px;"></div>
    <div class="section-label">Highlights Events</div>
    <div class="checkbox-group" id="custom-hl-events" style="margin-bottom:16px;"></div>
    <div style="display:flex; gap:8px; justify-content:flex-end;">
      <button class="btn" onclick="closeCustomDialog()">Cancel</button>
      <button class="btn btn-primary" onclick="confirmCustomReel()">Add Reel</button>
    </div>
  </div>
</div>

<script>
const API = '';
let eventTypes = [];
let jerseyColors = [];
let reelSpecs = [];

// ── Presets ──────────────────────────────────────────────────────────
const PRESETS = {
  keeper: {
    name: 'keeper',
    event_types: ['shot_stop_diving','shot_stop_standing','punch','catch','goal_kick',
                  'distribution_short','distribution_long','one_on_one','corner_kick','penalty']
  },
  highlights: {
    name: 'highlights',
    event_types: ['shot_on_target','shot_off_target','goal','near_miss','penalty','free_kick_shot']
  },
  saves: {
    name: 'saves',
    event_types: ['shot_stop_diving','shot_stop_standing','punch','catch','penalty']
  },
  distribution: {
    name: 'distribution',
    event_types: ['distribution_short','distribution_long','goal_kick']
  },
};

// ── Init ─────────────────────────────────────────────────────────────
async function init() {
  await Promise.all([loadEventTypes(), loadJerseyColors(), loadFiles()]);
  loadTeamDefaults();
  // Start with All GK preset
  addPresetReel('keeper');
  loadJobs();
  setInterval(loadJobs, 10000);
}

async function loadEventTypes() {
  try {
    const r = await fetch(API + '/meta/event-types');
    if (r.ok) eventTypes = await r.json();
  } catch (e) { /* fallback: empty */ }
}

async function loadJerseyColors() {
  try {
    const r = await fetch(API + '/meta/jersey-colors');
    if (!r.ok) return;
    jerseyColors = await r.json();
    for (const selId of ['team-outfield','team-gk','opp-outfield','opp-gk']) {
      const sel = document.getElementById(selId);
      sel.innerHTML = jerseyColors.map(c =>
        '<option value="' + c + '">' + c.replace(/_/g,' ') + '</option>'
      ).join('');
    }
    // Set reasonable defaults
    document.getElementById('team-outfield').value = 'white';
    document.getElementById('team-gk').value = 'neon_yellow';
    document.getElementById('opp-outfield').value = 'blue';
    document.getElementById('opp-gk').value = 'teal';
  } catch (e) { /* fallback */ }
}

function loadTeamDefaults() {
  // Try to load from saved team config
  fetch(API + '/team').then(r => {
    if (!r.ok) return;
    return r.json();
  }).then(cfg => {
    if (!cfg) return;
    document.getElementById('team-name').value = cfg.team_name || 'My Team';
    const kits = cfg.kits || {};
    const homeKit = kits['Home'] || kits[Object.keys(kits)[0]];
    if (homeKit) {
      document.getElementById('team-outfield').value = homeKit.outfield_color || 'white';
      document.getElementById('team-gk').value = homeKit.gk_color || 'neon_yellow';
    }
  }).catch(() => {});
}

async function loadFiles() {
  const sel = document.getElementById('nas-path');
  try {
    const r = await fetch(API + '/files');
    if (!r.ok) throw new Error(r.statusText);
    const files = await r.json();
    sel.innerHTML = '<option value="">-- select a video --</option>' +
      files.map(f => '<option value="' + f + '">' + f + '</option>').join('');
  } catch (e) {
    sel.innerHTML = '<option value="">(could not load files)</option>';
  }
}

// ── Reel builder ─────────────────────────────────────────────────────
function addPresetReel(presetKey) {
  const preset = PRESETS[presetKey];
  if (!preset) return;
  // Don't add duplicate
  if (reelSpecs.some(s => s.name === preset.name)) {
    showToast('Reel "' + preset.name + '" already added', true);
    return;
  }
  reelSpecs.push({...preset, event_types: [...preset.event_types]});
  renderReelCards();
}

function addCustomReel() {
  const dialog = document.getElementById('custom-reel-dialog');
  dialog.style.display = 'flex';
  document.getElementById('custom-reel-name').value = '';

  // Populate checkboxes
  const gkDiv = document.getElementById('custom-gk-events');
  const hlDiv = document.getElementById('custom-hl-events');
  gkDiv.innerHTML = '';
  hlDiv.innerHTML = '';
  for (const et of eventTypes) {
    const container = et.category === 'goalkeeper' ? gkDiv : hlDiv;
    const lbl = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = et.value;
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(' ' + et.label));
    container.appendChild(lbl);
  }
}

function closeCustomDialog() {
  document.getElementById('custom-reel-dialog').style.display = 'none';
}

function confirmCustomReel() {
  const name = document.getElementById('custom-reel-name').value.trim();
  if (!name) { showToast('Enter a reel name', true); return; }
  if (reelSpecs.some(s => s.name === name)) { showToast('Reel "' + name + '" already exists', true); return; }
  const checked = [];
  for (const cb of document.querySelectorAll('#custom-gk-events input:checked, #custom-hl-events input:checked')) {
    checked.push(cb.value);
  }
  if (!checked.length) { showToast('Select at least one event type', true); return; }
  reelSpecs.push({ name, event_types: checked });
  renderReelCards();
  closeCustomDialog();
}

function removeReel(idx) {
  reelSpecs.splice(idx, 1);
  renderReelCards();
}

function renderReelCards() {
  const container = document.getElementById('reel-cards');
  if (!reelSpecs.length) {
    container.innerHTML = '<span style="color:#8b949e;font-size:12px">No reels selected. Click a preset or add a custom reel.</span>';
    return;
  }
  const etMap = {};
  for (const et of eventTypes) etMap[et.value] = et;

  container.innerHTML = reelSpecs.map((spec, idx) => {
    const badges = spec.event_types.slice(0, 5).map(t => {
      const et = etMap[t];
      const cls = et && et.category === 'goalkeeper' ? 'gk' : 'hl';
      const label = et ? et.label : t;
      return '<span class="event-badge ' + cls + '">' + label + '</span>';
    }).join(' ');
    const more = spec.event_types.length > 5 ? ' <span class="event-badge">+' + (spec.event_types.length - 5) + '</span>' : '';
    return '<div class="reel-card">' +
      '<span class="reel-name">' + spec.name + '</span>' +
      '<span class="reel-types">' + badges + more + '</span>' +
      '<span class="remove-reel" onclick="removeReel(' + idx + ')" title="Remove">x</span>' +
      '</div>';
  }).join('');
}

// ── Jobs ─────────────────────────────────────────────────────────────
async function loadJobs() {
  try {
    const r = await fetch(API + '/jobs?limit=50');
    if (!r.ok) throw new Error(r.statusText);
    const jobs = await r.json();
    renderJobs(jobs);
    updateStats(jobs);
    document.getElementById('refresh-indicator').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    showToast('Failed to load jobs: ' + e.message, true);
  }
}

function renderJobs(jobs) {
  const tbody = document.getElementById('jobs-table-body');
  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No jobs yet. Submit a match above.</td></tr>';
    return;
  }
  const etMap = {};
  for (const et of eventTypes) etMap[et.value] = et;

  tbody.innerHTML = jobs.map(job => {
    const progress = job.progress_pct || 0;
    const dotClass = 'dot-' + job.status;
    const created = new Date(job.created_at).toLocaleString();
    const filename = job.video_file?.filename || '-';

    const produced = Object.keys(job.output_paths || {});
    // Show reel specs if available, otherwise fall back to reel_types
    const jobReels = (job.reels && job.reels.length) ? job.reels : (job.reel_types || []).map(rt => ({name: rt}));
    const reelHtml = produced.map(rt =>
      '<a class="reel-link" href="/reels/' + job.job_id + '/' + rt + '/download" target="_blank">' + rt + '</a>'
    ).join('');
    const pendingHtml = jobReels.filter(r => !produced.includes(r.name)).map(r => {
      if (job.status === 'complete') {
        return '<span style="color:#f85149;font-size:11px" title="No events detected">x ' + r.name + '</span>';
      }
      return '<span style="color:#8b949e;font-size:11px">' + r.name + '</span>';
    }).join(' ');

    const activeStatuses = new Set(['pending', 'ingesting', 'detecting', 'segmenting', 'assembling']);
    const pauseBtn = activeStatuses.has(job.status)
      ? '<button class="btn btn-sm btn-warn" onclick="pauseJob(\'' + job.job_id + '\')">Pause</button>'
      : '';
    const cancelBtn = (activeStatuses.has(job.status) || job.status === 'paused')
      ? '<button class="btn btn-sm btn-danger" onclick="cancelJob(\'' + job.job_id + '\')">Cancel</button>'
      : '';
    const resumeBtn = job.status === 'paused'
      ? '<button class="btn btn-sm" style="color:#3fb950;border-color:#3fb95033" onclick="resumeJob(\'' + job.job_id + '\')">Resume</button>'
      : '';
    const retryBtn = (job.status === 'failed' || job.status === 'cancelled')
      ? '<button class="btn btn-sm" onclick="retryJob(\'' + job.job_id + '\')">Retry</button>'
      : '';
    const deleteBtn = (job.status === 'failed' || job.status === 'complete' || job.status === 'paused' || job.status === 'cancelled')
      ? '<button class="btn btn-sm" style="color:#f85149;border-color:#f8514933" onclick="deleteJob(\'' + job.job_id + '\')">Delete</button>'
      : '';

    return '<tr>' +
      '<td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + filename + '">' + filename + '</td>' +
      '<td><span class="status"><span class="status-dot ' + dotClass + '"></span>' + job.status + '</span></td>' +
      '<td><div style="display:flex;align-items:center;gap:8px"><div class="progress-bar"><div class="progress-fill" style="width:' + progress + '%"></div></div><span style="font-size:12px;color:#8b949e">' + Math.round(progress) + '%</span></div></td>' +
      '<td><div class="reel-links">' + reelHtml + ' ' + pendingHtml + '</div></td>' +
      '<td style="font-size:12px;color:#8b949e">' + created + '</td>' +
      '<td style="display:flex;gap:4px">' + pauseBtn + resumeBtn + cancelBtn + retryBtn + deleteBtn + '</td>' +
      '</tr>';
  }).join('');
}

function updateStats(jobs) {
  const counts = { pending: 0, processing: 0, complete: 0, failed: 0 };
  const processingStatuses = new Set(['ingesting', 'detecting', 'segmenting', 'assembling']);
  for (const job of jobs) {
    if (job.status === 'pending' || job.status === 'paused') counts.pending++;
    else if (processingStatuses.has(job.status)) counts.processing++;
    else if (job.status === 'complete') counts.complete++;
    else if (job.status === 'failed' || job.status === 'cancelled') counts.failed++;
  }
  document.getElementById('stat-pending').textContent = counts.pending;
  document.getElementById('stat-processing').textContent = counts.processing;
  document.getElementById('stat-complete').textContent = counts.complete;
  document.getElementById('stat-failed').textContent = counts.failed;
}

async function submitJob() {
  const nasPath = document.getElementById('nas-path').value;
  if (!nasPath) { showToast('Please select a video file', true); return; }
  if (!reelSpecs.length) { showToast('Add at least one reel', true); return; }

  const gameStartMin = parseFloat(document.getElementById('game-start-min').value) || 0;
  const body = {
    nas_path: nasPath,
    match_config: {
      team: {
        team_name: document.getElementById('team-name').value || 'My Team',
        outfield_color: document.getElementById('team-outfield').value,
        gk_color: document.getElementById('team-gk').value,
      },
      opponent: {
        team_name: document.getElementById('opp-name').value || 'Opponent',
        outfield_color: document.getElementById('opp-outfield').value,
        gk_color: document.getElementById('opp-gk').value,
      },
    },
    reels: reelSpecs,
    game_start_sec: gameStartMin * 60,
  };

  try {
    const r = await fetch(API + '/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showToast('Job submitted: ' + data.job_id.slice(0, 8) + '...');
    document.getElementById('nas-path').selectedIndex = 0;
    loadJobs();
  } catch (e) {
    showToast('Submit failed: ' + e.message, true);
  }
}

async function retryJob(jobId) {
  try {
    const r = await fetch(API + '/jobs/' + jobId + '/retry', { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Job retried');
    loadJobs();
  } catch (e) { showToast('Retry failed: ' + e.message, true); }
}

async function deleteJob(jobId) {
  if (!confirm('Delete this job?')) return;
  try {
    const r = await fetch(API + '/jobs/' + jobId, { method: 'DELETE' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Job deleted');
    loadJobs();
  } catch (e) { showToast('Delete failed: ' + e.message, true); }
}

async function pauseJob(jobId) {
  try {
    const r = await fetch(API + '/jobs/' + jobId + '/pause', { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Pause requested');
    loadJobs();
  } catch (e) { showToast('Pause failed: ' + e.message, true); }
}

async function cancelJob(jobId) {
  if (!confirm('Cancel this job?')) return;
  try {
    const r = await fetch(API + '/jobs/' + jobId + '/cancel', { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Job cancelled');
    loadJobs();
  } catch (e) { showToast('Cancel failed: ' + e.message, true); }
}

async function resumeJob(jobId) {
  try {
    const r = await fetch(API + '/jobs/' + jobId + '/resume', { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Job resumed');
    loadJobs();
  } catch (e) { showToast('Resume failed: ' + e.message, true); }
}

function showToast(msg, isError = false) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isError ? ' error' : '');
  setTimeout(() => { t.className = 'toast'; }, 3500);
}

init();
</script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
@router.get("/ui/", response_class=HTMLResponse, include_in_schema=False)
async def monitoring_ui():
    """Embedded monitoring dashboard."""
    return HTMLResponse(_UI_HTML, headers={"Cache-Control": "no-store"})
