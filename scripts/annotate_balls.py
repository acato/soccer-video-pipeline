"""Browser-based ball annotation tool — single click to place bbox per frame.

Usage (on workstation with NAS access):
    pip install flask  # one-time
    python annotate_balls.py /path/to/yolo_ball_annotation
    # then open http://localhost:8848 in any browser

Workflow:
    - Click on the ball center → fixed-size bbox saved → auto-advance to next frame
    - Press "n" or click "Skip (no ball)" → frame marked as empty
    - Press "←" / "→" or click prev/next → navigate without saving
    - Press "u" / click Undo → re-edit current frame's label
    - Top progress bar shows N labeled / N skipped / N total

Output (YOLO format, class 0 = ball):
    {root}/labels/{idx:04d}.txt   # "0 cx cy w h" per line, normalized
    {root}/labels/{idx:04d}.skip   # marker when explicitly skipped (no ball)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from flask import Flask, jsonify, request, send_file, Response
except ImportError:
    print("ERROR: pip install flask", file=sys.stderr)
    sys.exit(1)

# ── Default ball bbox size (fraction of frame width/height) ──
# Tuned for 1920×1080 frames where the ball is typically 8-25 px wide;
# user can override via the size slider in the UI.
DEFAULT_BBOX_SIZE = 0.012   # ~23 px on 1920w


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Ball Annotator</title>
<style>
  body { margin: 0; background: #222; color: #ddd; font-family: -apple-system, sans-serif; }
  #header { padding: 8px 16px; background: #111; display: flex; gap: 16px;
            align-items: center; flex-wrap: wrap; }
  #progress { font-size: 14px; }
  #progress b { color: #fff; }
  #info { font-size: 12px; color: #888; }
  button { padding: 6px 12px; background: #333; color: #ddd; border: 1px solid #555;
           border-radius: 3px; cursor: pointer; }
  button:hover { background: #444; }
  button.primary { background: #2a5; border-color: #4a7; }
  button.skip { background: #aa4; border-color: #cc6; color: #000; }
  #wrap { position: relative; display: inline-block; user-select: none; }
  #frame { display: block; max-width: 100%; cursor: crosshair; }
  #bbox { position: absolute; border: 2px solid #f00; pointer-events: none;
          box-shadow: 0 0 0 2px rgba(0,0,0,0.5); }
  .row { display: flex; gap: 12px; align-items: center; }
  input[type=range] { width: 120px; }
  #empty-msg { padding: 30px; text-align: center; }
</style>
</head>
<body>
<div id="header">
  <div id="progress">Frame <b id="cur">…</b> / <b id="total">…</b> · labeled <b id="n_labeled">0</b> · skipped <b id="n_skip">0</b> · pending <b id="n_pending">0</b></div>
  <div class="row"><button onclick="back()">← Back (B)</button></div>
  <div class="row">bbox size <input type="range" id="size" min="0.005" max="0.04" step="0.001" value="0.012"/> <span id="sizeval">0.012</span></div>
  <div class="row"><button class="skip" onclick="skip()">No ball / Skip (N)</button></div>
  <div class="row"><button class="primary" onclick="goToFirstUnlabeled()">Jump to next unlabeled (J)</button></div>
  <div id="info">click on the ball center · arrow keys navigate · U to undo current label</div>
</div>
<div style="padding: 16px;">
  <div id="wrap">
    <img id="frame" src=""/>
    <div id="bbox" style="display:none;"></div>
  </div>
  <div id="empty-msg" style="display:none;">No frames found.</div>
</div>
<script>
let cur = 0, total = 0, manifest = [], labelStatus = {};
let bboxSize = 0.012;
let lastClick = null;  // {x_n, y_n}

document.getElementById("size").oninput = e => {
  bboxSize = parseFloat(e.target.value);
  document.getElementById("sizeval").textContent = bboxSize.toFixed(3);
  if (lastClick) drawBboxAt(lastClick.x_n, lastClick.y_n);
};

async function loadState() {
  const r = await fetch("/state").then(r => r.json());
  total = r.total;
  manifest = r.manifest;
  labelStatus = r.labels;
  document.getElementById("total").textContent = total;
  if (total === 0) {
    document.getElementById("wrap").style.display = "none";
    document.getElementById("empty-msg").style.display = "block";
    return;
  }
  cur = r.resume_idx;
  showFrame();
  updateCounts();
}

function updateCounts() {
  let nL = 0, nS = 0;
  for (const k in labelStatus) {
    if (labelStatus[k] === "labeled") nL++;
    else if (labelStatus[k] === "skipped") nS++;
  }
  document.getElementById("n_labeled").textContent = nL;
  document.getElementById("n_skip").textContent = nS;
  document.getElementById("n_pending").textContent = total - nL - nS;
}

function showFrame() {
  document.getElementById("cur").textContent = cur;
  const img = document.getElementById("frame");
  img.src = `/img/${cur}?v=${Date.now()}`;
  document.getElementById("bbox").style.display = "none";
  lastClick = null;
  // Show existing label if present
  fetch(`/label/${cur}`).then(r => r.json()).then(d => {
    if (d.bbox) {
      lastClick = {x_n: d.bbox.cx, y_n: d.bbox.cy};
      bboxSize = d.bbox.w;
      document.getElementById("size").value = bboxSize;
      document.getElementById("sizeval").textContent = bboxSize.toFixed(3);
      // Wait for image to load before drawing
      img.onload = () => drawBboxAt(d.bbox.cx, d.bbox.cy);
      if (img.complete) drawBboxAt(d.bbox.cx, d.bbox.cy);
    }
  });
}

function drawBboxAt(x_n, y_n) {
  const img = document.getElementById("frame");
  const r = img.getBoundingClientRect();
  const w_disp = img.clientWidth;
  const h_disp = img.clientHeight;
  const cx = x_n * w_disp;
  const cy = y_n * h_disp;
  const bw = bboxSize * w_disp;
  const bh = bboxSize * h_disp;
  const bbox = document.getElementById("bbox");
  bbox.style.left = (cx - bw / 2) + "px";
  bbox.style.top = (cy - bh / 2) + "px";
  bbox.style.width = bw + "px";
  bbox.style.height = bh + "px";
  bbox.style.display = "block";
}

document.getElementById("frame").addEventListener("click", e => {
  const img = e.target;
  const r = img.getBoundingClientRect();
  const x_n = (e.clientX - r.left) / r.width;
  const y_n = (e.clientY - r.top) / r.height;
  lastClick = {x_n, y_n};
  drawBboxAt(x_n, y_n);
  // Save and advance
  saveBbox(x_n, y_n).then(() => {
    if (cur < total - 1) { cur++; showFrame(); }
  });
});

async function saveBbox(x_n, y_n) {
  const r = await fetch(`/label/${cur}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({cx: x_n, cy: y_n, w: bboxSize, h: bboxSize}),
  });
  const j = await r.json();
  labelStatus[cur] = "labeled";
  updateCounts();
}

async function skip() {
  const r = await fetch(`/skip/${cur}`, {method: "POST"});
  labelStatus[cur] = "skipped";
  updateCounts();
  if (cur < total - 1) { cur++; showFrame(); }
}

async function back() {
  if (cur > 0) { cur--; showFrame(); }
}

async function next_() {
  if (cur < total - 1) { cur++; showFrame(); }
}

async function undo() {
  await fetch(`/label/${cur}`, {method: "DELETE"});
  delete labelStatus[cur];
  updateCounts();
  showFrame();
}

async function goToFirstUnlabeled() {
  const r = await fetch("/state").then(r => r.json());
  labelStatus = r.labels;
  updateCounts();
  for (let i = 0; i < total; i++) {
    if (!(i in labelStatus)) { cur = i; showFrame(); return; }
  }
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "n" || e.key === " ") { e.preventDefault(); skip(); }
  else if (e.key === "ArrowRight") next_();
  else if (e.key === "ArrowLeft" || e.key === "b") back();
  else if (e.key === "u") undo();
  else if (e.key === "j") goToFirstUnlabeled();
});

loadState();
</script>
</body>
</html>
"""


def make_app(root: Path) -> Flask:
    app = Flask(__name__)
    frames_dir = root / "frames"
    labels_dir = root / "labels"
    labels_dir.mkdir(exist_ok=True)
    if not frames_dir.exists():
        sys.exit(f"frames dir not found: {frames_dir}")

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    print(f"  loaded {len(frame_paths)} frames from {frames_dir}")

    def label_status(idx: int) -> str | None:
        if (labels_dir / f"{idx:04d}.txt").exists():
            return "labeled"
        if (labels_dir / f"{idx:04d}.skip").exists():
            return "skipped"
        return None

    @app.get("/")
    def index():
        return Response(HTML, mimetype="text/html")

    @app.get("/img/<int:idx>")
    def img(idx: int):
        if idx < 0 or idx >= len(frame_paths):
            return ("not found", 404)
        return send_file(str(frame_paths[idx]), mimetype="image/jpeg")

    @app.get("/state")
    def state():
        labels = {i: s for i in range(len(frame_paths))
                  if (s := label_status(i)) is not None}
        # Resume at first unlabeled
        resume = next((i for i in range(len(frame_paths)) if i not in labels), 0)
        manifest = []
        mf = root / "manifest.jsonl"
        if mf.exists():
            for line in mf.read_text().splitlines():
                if line.strip():
                    manifest.append(json.loads(line))
        return jsonify({"total": len(frame_paths), "labels": labels,
                        "resume_idx": resume, "manifest": manifest})

    @app.get("/label/<int:idx>")
    def get_label(idx: int):
        f = labels_dir / f"{idx:04d}.txt"
        if not f.exists():
            return jsonify({"bbox": None})
        line = f.read_text().strip().split()
        if len(line) >= 5:
            return jsonify({"bbox": {"cx": float(line[1]), "cy": float(line[2]),
                                     "w": float(line[3]), "h": float(line[4])}})
        return jsonify({"bbox": None})

    @app.post("/label/<int:idx>")
    def post_label(idx: int):
        d = request.get_json() or {}
        cx, cy = float(d.get("cx", 0)), float(d.get("cy", 0))
        w, h = float(d.get("w", DEFAULT_BBOX_SIZE)), float(d.get("h", DEFAULT_BBOX_SIZE))
        # Save YOLO format: <class> cx cy w h
        (labels_dir / f"{idx:04d}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        # Drop any existing skip marker
        (labels_dir / f"{idx:04d}.skip").unlink(missing_ok=True)
        return jsonify({"ok": True})

    @app.delete("/label/<int:idx>")
    def del_label(idx: int):
        (labels_dir / f"{idx:04d}.txt").unlink(missing_ok=True)
        (labels_dir / f"{idx:04d}.skip").unlink(missing_ok=True)
        return jsonify({"ok": True})

    @app.post("/skip/<int:idx>")
    def post_skip(idx: int):
        (labels_dir / f"{idx:04d}.skip").touch()
        (labels_dir / f"{idx:04d}.txt").unlink(missing_ok=True)
        return jsonify({"ok": True})

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Annotation workspace root (must contain frames/)")
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"workspace not found: {root}")

    app = make_app(root)
    print(f"\n  Open http://{args.host}:{args.port} in your browser")
    print(f"  Labels save to {root}/labels/")
    print(f"  Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
