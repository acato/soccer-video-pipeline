"""Build keeper + highlights reels from tiered events.

For each game produces two MP4s in --out-dir:
  <game>_keeper.mp4      = goals (any tier) + saves (any tier)
  <game>_highlights.mp4  = shots (any tier)

Clip windows: per-event pre/post pad, then merge overlapping windows.
Output is stream-copied (codec=copy) — no re-encode, fast and lossless.

Usage:
  python scripts/build_reels.py --out-dir /Volumes/transit/reels [--games game_20 ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GAMES = {
    "game_20": {
        "source": "/Users/aless/soccer-working/2026-04-18 Celtic - Reign GA 11.mp4",
        "tiered": "/tmp/kickoff_game_20_tiered_events.jsonl",
        "label": "Celtic vs Reign 2026-04-18",
    },
    "game_21": {
        "source": "/Users/aless/soccer-working/2026-04-25 Eastern WA Surf - Reign GA11.mp4",
        "tiered": "/tmp/kickoff_game_21_tiered_events.jsonl",
        "label": "Eastern WA Surf vs Reign 2026-04-25",
    },
    "game_22": {
        "source": "/Users/aless/soccer-working/2026-04-26 Spokane Shadow - Reign GA11.mp4",
        "tiered": "/tmp/kickoff_game_22_tiered_events.jsonl",
        "label": "Spokane Shadow vs Reign 2026-04-26",
    },
    "rush": {
        "source": "/Users/aless/soccer-working/2026-02-07 - Rush - GA2008.mp4",
        "tiered": "/tmp/kickoff_rush_tiered_events.jsonl",
        "label": "Rush vs GA 2026-02-07",
    },
}

PRE_PAD_SEC = 6.0
POST_PAD_SEC = 12.0
MERGE_GAP_SEC = 3.0
MAX_CLIP_SEC = 60.0


def _t(e):
    return e.get("start_sec", e.get("timestamp_start"))


def select_events(events: list[dict], reel: str) -> list[dict]:
    out = []
    for e in events:
        md = e.get("metadata", {}) or {}
        if reel == "keeper":
            if md.get("goal_tier") or md.get("save_tier"):
                out.append(e)
        elif reel == "highlights":
            if md.get("shot_tier"):
                out.append(e)
    return out


def build_clips(events: list[dict], duration: float | None = None) -> list[tuple[float, float, list[dict]]]:
    """Apply padding, sort, merge overlapping windows. Returns (start, end, events)."""
    windows = []
    for e in events:
        t = _t(e)
        if t is None:
            continue
        start = max(0.0, t - PRE_PAD_SEC)
        end = t + POST_PAD_SEC
        if duration is not None:
            end = min(duration, end)
        windows.append((start, end, e))
    windows.sort(key=lambda x: x[0])

    merged: list[tuple[float, float, list[dict]]] = []
    for start, end, e in windows:
        if merged and start - merged[-1][1] <= MERGE_GAP_SEC:
            prev_s, prev_e, prev_evs = merged[-1]
            new_end = max(prev_e, end)
            if new_end - prev_s <= MAX_CLIP_SEC:
                merged[-1] = (prev_s, new_end, prev_evs + [e])
                continue
            # Trim to avoid overlap if cap forced no-merge
            safe_start = max(start, prev_e)
            if safe_start < end:
                merged.append((safe_start, end, [e]))
        else:
            merged.append((start, end, [e]))
    return merged


def probe_duration(ffmpeg_bin: str, src: Path) -> float | None:
    """Use ffmpeg to get duration in seconds. Returns None on failure."""
    # Use ffmpeg with -v error -i ... -f null - to get duration from stderr.
    # Cheaper: ffprobe — assume it's next to ffmpeg.
    ffprobe = Path(ffmpeg_bin).with_name("ffprobe")
    if not ffprobe.exists():
        return None
    r = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def extract_clip(ffmpeg_bin: str, src: Path, start: float, end: float, out: Path) -> bool:
    """Extract a single clip via stream copy."""
    duration = max(0.1, end - start)
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    EXTRACT FAIL [{start:.0f}-{end:.0f}]: {r.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


def concat_clips(ffmpeg_bin: str, clip_paths: list[Path], out: Path) -> bool:
    """Use concat demuxer to join clips with stream copy."""
    if not clip_paths:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        listfile = Path(f.name)
        for p in clip_paths:
            f.write(f"file '{p.as_posix()}'\n")
    try:
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  CONCAT FAIL: {r.stderr.strip()[:400]}", file=sys.stderr)
            return False
        return True
    finally:
        listfile.unlink(missing_ok=True)


def build_reel(ffmpeg_bin: str, game: str, reel: str, out_dir: Path) -> Path | None:
    cfg = GAMES[game]
    src = Path(cfg["source"])
    tiered = Path(cfg["tiered"])
    if not src.exists():
        print(f"  SKIP {game}/{reel}: source missing {src}", file=sys.stderr)
        return None
    if not tiered.exists():
        print(f"  SKIP {game}/{reel}: tiered missing {tiered}", file=sys.stderr)
        return None

    events = [json.loads(l) for l in tiered.read_text().splitlines() if l.strip()]
    chosen = select_events(events, reel)
    duration = probe_duration(ffmpeg_bin, src)
    clips = build_clips(chosen, duration=duration)
    if not clips:
        print(f"  {game}/{reel}: 0 events → no reel")
        return None

    total = sum(end - start for start, end, _ in clips)
    print(f"  {game}/{reel}: {len(chosen)} events → {len(clips)} clips ({total:.0f}s)")

    out_path = out_dir / f"{game}_{reel}.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"reel_{game}_{reel}_") as tmp:
        tmp_dir = Path(tmp)
        clip_paths: list[Path] = []
        for i, (start, end, _) in enumerate(clips):
            cp = tmp_dir / f"clip_{i:04d}.mp4"
            if extract_clip(ffmpeg_bin, src, start, end, cp):
                clip_paths.append(cp)
        if not clip_paths:
            print(f"  {game}/{reel}: all extracts failed")
            return None
        if not concat_clips(ffmpeg_bin, clip_paths, out_path):
            return None
    print(f"  → wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--games", nargs="*", default=list(GAMES.keys()))
    ap.add_argument("--reels", nargs="*", default=["keeper", "highlights"])
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")
    args = ap.parse_args()

    if not Path(args.ffmpeg).exists():
        print(f"ERROR: ffmpeg not found at {args.ffmpeg}", file=sys.stderr)
        sys.exit(1)

    print(f"output dir: {args.out_dir}")
    print(f"ffmpeg:     {args.ffmpeg}")
    print()
    for game in args.games:
        if game not in GAMES:
            print(f"  unknown game: {game}", file=sys.stderr); continue
        print(f"=== {game} ({GAMES[game]['label']}) ===")
        for reel in args.reels:
            build_reel(args.ffmpeg, game, reel, args.out_dir)


if __name__ == "__main__":
    main()
