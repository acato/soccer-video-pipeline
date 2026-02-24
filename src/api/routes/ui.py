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
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .progress-bar { width: 80px; height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; }
  .progress-fill { height: 100%; background: #388bfd; border-radius: 3px; transition: width 0.5s; }
  .btn { padding: 5px 12px; border-radius: 6px; border: 1px solid #30363d; cursor: pointer;
         font-size: 12px; background: #21262d; color: #e1e4e8; transition: background 0.15s; }
  .btn:hover { background: #30363d; }
  .btn-sm { padding: 3px 8px; font-size: 11px; }
  .reel-links { display: flex; gap: 6px; }
  .reel-link { color: #388bfd; text-decoration: none; font-size: 12px;
               padding: 2px 6px; border: 1px solid #388bfd33; border-radius: 4px; }
  .reel-link:hover { background: #388bfd22; }
  .empty-state { padding: 48px 16px; text-align: center; color: #8b949e; }
  .submit-form { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 20px; margin-bottom: 24px; }
  .submit-form h2 { font-size: 15px; font-weight: 600; margin-bottom: 16px; }
  .form-row { display: flex; gap: 12px; align-items: flex-end; }
  .form-group { flex: 1; }
  label { display: block; font-size: 12px; color: #8b949e; margin-bottom: 6px; }
  input[type=text] { width: 100%; padding: 8px 12px; background: #0d1117; border: 1px solid #30363d;
                     border-radius: 6px; color: #e1e4e8; font-size: 14px; }
  input[type=text]:focus { outline: none; border-color: #388bfd; }
  .btn-primary { background: #238636; border-color: #238636; color: white; padding: 8px 16px; }
  .btn-primary:hover { background: #2ea043; }
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
  <div class="submit-form">
    <h2>Submit Match</h2>
    <div class="form-row">
      <div class="form-group">
        <label>NAS Path (relative to mount)</label>
        <input type="text" id="nas-path" placeholder="matches/2025_01_15_game.mp4" />
      </div>
      <div class="form-group" style="flex:0 0 180px">
        <label>Reel Types</label>
        <input type="text" id="reel-types" value="goalkeeper,highlights" />
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
          <th>Job ID</th>
          <th>File</th>
          <th>Status</th>
          <th>Progress</th>
          <th>Reels</th>
          <th>Created</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody id="jobs-table-body">
        <tr><td colspan="7" class="empty-state">Loading...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const API = '';  // Same origin

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
    tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No jobs yet. Submit a match above.</td></tr>';
    return;
  }
  tbody.innerHTML = jobs.map(job => {
    const progress = job.progress_pct || 0;
    const statusClass = job.status.replace('_', '-');
    const dotClass = 'dot-' + job.status;
    const created = new Date(job.created_at).toLocaleString();
    const shortId = job.job_id.slice(0, 8);
    const filename = job.video_file?.filename || '—';

    const reelLinks = (job.reel_types || []).map(rt => {
      if (job.output_paths && job.output_paths[rt]) {
        return `<a class="reel-link" href="/reels/${job.job_id}/${rt}/download" target="_blank">⬇ ${rt}</a>`;
      }
      return `<span style="color:#8b949e;font-size:12px">${rt}</span>`;
    }).join('');

    const retryBtn = job.status === 'failed'
      ? `<button class="btn btn-sm" onclick="retryJob('${job.job_id}')">↩ Retry</button>`
      : '';
    const deleteBtn = (job.status === 'failed' || job.status === 'complete')
      ? `<button class="btn btn-sm" style="color:#f85149;border-color:#f8514933" onclick="deleteJob('${job.job_id}')">✕ Delete</button>`
      : '';

    return `<tr>
      <td style="font-family:monospace;font-size:12px" title="${job.job_id}">${shortId}…</td>
      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${filename}">${filename}</td>
      <td><span class="status"><span class="status-dot ${dotClass}"></span>${job.status}</span></td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="progress-bar"><div class="progress-fill" style="width:${progress}%"></div></div>
          <span style="font-size:12px;color:#8b949e">${Math.round(progress)}%</span>
        </div>
      </td>
      <td><div class="reel-links">${reelLinks}</div></td>
      <td style="font-size:12px;color:#8b949e">${created}</td>
      <td style="display:flex;gap:4px">${retryBtn}${deleteBtn}</td>
    </tr>`;
  }).join('');
}

function updateStats(jobs) {
  const counts = { pending: 0, processing: 0, complete: 0, failed: 0 };
  const processingStatuses = new Set(['ingesting', 'detecting', 'segmenting', 'assembling']);
  for (const job of jobs) {
    if (job.status === 'pending') counts.pending++;
    else if (processingStatuses.has(job.status)) counts.processing++;
    else if (job.status === 'complete') counts.complete++;
    else if (job.status === 'failed') counts.failed++;
  }
  document.getElementById('stat-pending').textContent = counts.pending;
  document.getElementById('stat-processing').textContent = counts.processing;
  document.getElementById('stat-complete').textContent = counts.complete;
  document.getElementById('stat-failed').textContent = counts.failed;
}

async function submitJob() {
  const nasPath = document.getElementById('nas-path').value.trim();
  if (!nasPath) { showToast('Please enter a NAS path', true); return; }
  const reelTypes = document.getElementById('reel-types').value.split(',').map(s => s.trim());
  try {
    const r = await fetch(API + '/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nas_path: nasPath, reel_types: reelTypes }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showToast('Job submitted: ' + data.job_id.slice(0, 8) + '…');
    document.getElementById('nas-path').value = '';
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

function showToast(msg, isError = false) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isError ? ' error' : '');
  setTimeout(() => { t.className = 'toast'; }, 3500);
}

// Auto-refresh every 10s
loadJobs();
setInterval(loadJobs, 10000);
</script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def monitoring_ui():
    """Embedded monitoring dashboard."""
    return HTMLResponse(_UI_HTML, headers={"Cache-Control": "no-store"})
