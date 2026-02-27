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
  .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
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
  .reel-links { display: flex; gap: 6px; }
  .reel-link { color: #388bfd; text-decoration: none; font-size: 12px;
               padding: 2px 6px; border: 1px solid #388bfd33; border-radius: 4px; }
  .reel-link:hover { background: #388bfd22; }
  .empty-state { padding: 48px 16px; text-align: center; color: #8b949e; }
  .submit-form { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 20px; margin-bottom: 24px; }
  .submit-form h2 { font-size: 15px; font-weight: 600; margin-bottom: 16px; }
  .form-row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
  .form-group { flex: 1; }
  label { display: block; font-size: 12px; color: #8b949e; margin-bottom: 6px; }
  input[type=text], input[type=number] { width: 100%; padding: 8px 12px; background: #0d1117; border: 1px solid #30363d;
                     border-radius: 6px; color: #e1e4e8; font-size: 14px; }
  input[type=text]:focus, input[type=number]:focus { outline: none; border-color: #388bfd; }
  select { width: 100%; padding: 8px 12px; background: #0d1117; border: 1px solid #30363d;
           border-radius: 6px; color: #e1e4e8; font-size: 14px; }
  select:focus { outline: none; border-color: #388bfd; }
  .checkbox-group { display: flex; gap: 12px; align-items: center; height: 36px; }
  .checkbox-group label { display: flex; align-items: center; gap: 6px; margin-bottom: 0;
                          font-size: 14px; color: #e1e4e8; cursor: pointer; }
  .checkbox-group input[type=checkbox] { accent-color: #388bfd; width: 16px; height: 16px; cursor: pointer; }
  .btn-primary { background: #238636; border-color: #238636; color: white; padding: 8px 16px; }
  .btn-primary:hover { background: #2ea043; }
  .team-banner { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 16px 20px; margin-bottom: 24px; display: flex; align-items: center; gap: 12px; }
  .team-banner .team-name { font-size: 16px; font-weight: 600; }
  .team-banner .team-hint { font-size: 12px; color: #8b949e; }
  .no-team-msg { background: #161b22; border: 1px solid #d29922; border-radius: 8px;
                 padding: 16px 20px; margin-bottom: 24px; color: #d29922; font-size: 14px; }
  .toast { position: fixed; bottom: 24px; right: 24px; background: #238636; color: white;
           padding: 12px 16px; border-radius: 8px; font-size: 14px; opacity: 0; transition: opacity 0.3s;
           max-width: 320px; }
  .toast.show { opacity: 1; }
  .toast.error { background: #da3633; }
  #refresh-indicator { font-size: 12px; color: #8b949e; }
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
  <div id="team-banner-area"></div>

  <div class="submit-form" id="submit-form-area">
    <h2>Submit Match</h2>
    <div class="form-row">
      <div class="form-group">
        <label>Video File</label>
        <select id="nas-path">
          <option value="">Loading files...</option>
        </select>
      </div>
      <div class="form-group" style="flex:0 0 160px">
        <label>Jersey</label>
        <select id="kit-select">
          <option value="">Loading...</option>
        </select>
      </div>
      <div class="form-group" style="flex:0 0 200px">
        <label>Reels</label>
        <div class="checkbox-group">
          <label><input type="checkbox" id="reel-gk" checked /> Goalkeeper</label>
          <label><input type="checkbox" id="reel-hl" checked /> Highlights</label>
        </div>
      </div>
      <div class="form-group" style="flex:0 0 120px">
        <label>Game start (min)</label>
        <input type="number" id="game-start-min" value="0" min="0" step="0.5" />
      </div>
      <button class="btn btn-primary" onclick="submitJob()">Submit Job</button>
    </div>
  </div>

  <div class="stats-row">
    <div class="stat-card pending"><div class="value" id="stat-pending">—</div><div class="label">Pending</div></div>
    <div class="stat-card processing"><div class="value" id="stat-processing">—</div><div class="label">Processing</div></div>
    <div class="stat-card complete"><div class="value" id="stat-complete">—</div><div class="label">Complete</div></div>
    <div class="stat-card failed"><div class="value" id="stat-failed">—</div><div class="label">Failed</div></div>
  </div>

  <div class="panel">
    <div class="panel-header">
      <h2>Jobs</h2>
      <button class="btn btn-sm" onclick="loadJobs()">↻ Refresh</button>
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

<script>
const API = '';  // Same origin
let teamConfig = null;

async function loadTeamConfig() {
  const banner = document.getElementById('team-banner-area');
  const kitSelect = document.getElementById('kit-select');
  try {
    const r = await fetch(API + '/team');
    if (r.status === 404) {
      teamConfig = null;
      banner.innerHTML = '<div class="no-team-msg">Set up your team first! In your terminal, run: <code>./setup-team.sh "Your Team" --kit Home blue teal</code></div>';
      kitSelect.innerHTML = '<option value="">(no team set up)</option>';
      return;
    }
    if (!r.ok) throw new Error(r.statusText);
    teamConfig = await r.json();
    const kits = Object.keys(teamConfig.kits || {});
    banner.innerHTML = '<div class="team-banner"><span class="team-name">' + (teamConfig.team_name || 'My Team') +
      '</span><span class="team-hint">' + kits.length + ' jersey' + (kits.length !== 1 ? 's' : '') + ' saved</span></div>';
    kitSelect.innerHTML = kits.map(k => '<option value="' + k + '">' + k + '</option>').join('');
    if (!kits.length) {
      kitSelect.innerHTML = '<option value="">(no jerseys)</option>';
    }
  } catch (e) {
    banner.innerHTML = '';
    kitSelect.innerHTML = '<option value="">Home</option>';
  }
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
  tbody.innerHTML = jobs.map(job => {
    const progress = job.progress_pct || 0;
    const statusClass = job.status.replace('_', '-');
    const dotClass = 'dot-' + job.status;
    const created = new Date(job.created_at).toLocaleString();
    const filename = job.video_file?.filename || '—';

    const produced = Object.keys(job.output_paths || {});
    const requested = job.reel_types || [];
    const missing = requested.filter(rt => !produced.includes(rt));
    const reelLinks = produced.map(rt =>
      `<a class="reel-link" href="/reels/${job.job_id}/${rt}/download" target="_blank">⬇ ${rt}</a>`
    ).join('') + (missing.length && job.status === 'complete'
      ? missing.map(rt => `<span style="color:#f85149;font-size:11px" title="No ${rt} events detected">✕ ${rt}</span>`).join('')
      : missing.map(rt => `<span style="color:#8b949e;font-size:11px">${rt}</span>`).join(''));

    const activeStatuses = new Set(['pending', 'ingesting', 'detecting', 'segmenting', 'assembling']);
    const pauseBtn = activeStatuses.has(job.status)
      ? `<button class="btn btn-sm btn-warn" onclick="pauseJob('${job.job_id}')">⏸ Pause</button>`
      : '';
    const cancelBtn = (activeStatuses.has(job.status) || job.status === 'paused')
      ? `<button class="btn btn-sm btn-danger" onclick="cancelJob('${job.job_id}')">✕ Cancel</button>`
      : '';
    const resumeBtn = job.status === 'paused'
      ? `<button class="btn btn-sm" style="color:#3fb950;border-color:#3fb95033" onclick="resumeJob('${job.job_id}')">▶ Resume</button>`
      : '';
    const retryBtn = (job.status === 'failed' || job.status === 'cancelled')
      ? `<button class="btn btn-sm" onclick="retryJob('${job.job_id}')">↩ Retry</button>`
      : '';
    const deleteBtn = (job.status === 'failed' || job.status === 'complete' || job.status === 'paused' || job.status === 'cancelled')
      ? `<button class="btn btn-sm" style="color:#f85149;border-color:#f8514933" onclick="deleteJob('${job.job_id}')">✕ Delete</button>`
      : '';

    return `<tr>
      <td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${filename}">${filename}</td>
      <td><span class="status"><span class="status-dot ${dotClass}"></span>${job.status}</span></td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="progress-bar"><div class="progress-fill" style="width:${progress}%"></div></div>
          <span style="font-size:12px;color:#8b949e">${Math.round(progress)}%</span>
        </div>
      </td>
      <td><div class="reel-links">${reelLinks}</div></td>
      <td style="font-size:12px;color:#8b949e">${created}</td>
      <td style="display:flex;gap:4px">${pauseBtn}${resumeBtn}${cancelBtn}${retryBtn}${deleteBtn}</td>
    </tr>`;
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
  const kitSelect = document.getElementById('kit-select');
  const kitName = kitSelect.value;
  const reelTypes = [];
  if (document.getElementById('reel-gk').checked) reelTypes.push('keeper');
  if (document.getElementById('reel-hl').checked) reelTypes.push('highlights');
  if (!reelTypes.length) { showToast('Select at least one reel type', true); return; }
  const gameStartMin = parseFloat(document.getElementById('game-start-min').value) || 0;
  const body = { nas_path: nasPath, reel_types: reelTypes, game_start_sec: gameStartMin * 60 };
  if (kitName) body.kit_name = kitName;
  try {
    const r = await fetch(API + '/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showToast('Job submitted: ' + data.job_id.slice(0, 8) + '…');
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
  } catch (e) {
    showToast('Retry failed: ' + e.message, true);
  }
}

async function deleteJob(jobId) {
  if (!confirm('Delete this job?')) return;
  try {
    const r = await fetch(API + '/jobs/' + jobId, { method: 'DELETE' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Job deleted');
    loadJobs();
  } catch (e) {
    showToast('Delete failed: ' + e.message, true);
  }
}

async function pauseJob(jobId) {
  try {
    const r = await fetch(API + '/jobs/' + jobId + '/pause', { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Pause requested');
    loadJobs();
  } catch (e) {
    showToast('Pause failed: ' + e.message, true);
  }
}

async function cancelJob(jobId) {
  if (!confirm('Cancel this job? This cannot be undone.')) return;
  try {
    const r = await fetch(API + '/jobs/' + jobId + '/cancel', { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Job cancelled');
    loadJobs();
  } catch (e) {
    showToast('Cancel failed: ' + e.message, true);
  }
}

async function resumeJob(jobId) {
  try {
    const r = await fetch(API + '/jobs/' + jobId + '/resume', { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Job resumed');
    loadJobs();
  } catch (e) {
    showToast('Resume failed: ' + e.message, true);
  }
}

function showToast(msg, isError = false) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isError ? ' error' : '');
  setTimeout(() => { t.className = 'toast'; }, 3500);
}

// Load configs once, then auto-refresh jobs every 10s
loadTeamConfig();
loadFiles();
loadJobs();
setInterval(loadJobs, 10000);
</script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
@router.get("/ui/", response_class=HTMLResponse, include_in_schema=False)
async def monitoring_ui():
    """Embedded monitoring dashboard."""
    return HTMLResponse(_UI_HTML, headers={"Cache-Control": "no-store"})
