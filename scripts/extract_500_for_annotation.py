"""Extract 500 frames across the 4 new-venue games for manual ball annotation.

Mix per game (~125 frames each):
  - 60 frames at GT event timestamps (ball is in play/visible by definition)
  - 65 frames at random play timestamps (covers regular gameplay diversity)

Output to /mnt/transit/soccer-finetune/yolo_ball_annotation/frames/000000.jpg ...
plus manifest.jsonl mapping idx → game_id, video_ts, source_event.

Run on llm:
    ~/quant-env/bin/python ~/extract_500_for_annotation.py
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/usr/bin/ffprobe"

OUT_ROOT = Path("/mnt/transit/soccer-finetune/yolo_ball_annotation")

FRAMES_PER_GAME = 125
GT_FRAMES_PER_GAME = 60
RANDOM_FRAMES_PER_GAME = 65
RANDOM_SEED = 42

GAMES = {
    "game_19": dict(
        video="/mnt/transit/Games/19/1773631328983_seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-1aa48cf7-b31f-41c9-aa17-cef4b4c22267-1773632512.672041-encoded.mp4",
        gt_h1="/mnt/transit/Games/19/2026-03-15_Seattle Reign 2011 GA (U15) vs Oregon Premier FC U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/19/2026-03-15_Seattle Reign 2011 GA (U15) vs Oregon Premier FC U15 (W)_2nd Half.json",
        video_offset=0.0, half2_video_start=2863.0, half2_game_offset=2400.0,
    ),
    "game_20": dict(
        video="/mnt/transit/Games/20/2026-04-18 Celtic - Reign GA 11.mp4",
        gt_h1="/mnt/transit/Games/20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_2nd Half.json",
        video_offset=124.0, half2_video_start=3554.0, half2_game_offset=2400.0,
    ),
    "game_21": dict(
        video="/mnt/transit/Games/21/2026-04-25 Eastern WA Surf - Reign GA11.mp4",
        gt_h1="/mnt/transit/Games/21/2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/21/2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_2nd Half.json",
        video_offset=250.0, half2_video_start=4130.0, half2_game_offset=2400.0,
    ),
    "game_22": dict(
        video="/mnt/transit/Games/22/2026-04-26 Spokane Shadow - Reign GA11.mp4",
        gt_h1="/mnt/transit/Games/22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_2nd Half.json",
        video_offset=90.0, half2_video_start=2900.0, half2_game_offset=2700.0,
    ),
}

# Prefer events where the ball is in motion / clearly in play
PREFERRED_EVENTS = {
    "Shots & Goals", "Saves/Catches", "Saves/Parries", "Goals Conceded",
    "Set Pieces/Corners", "Set Pieces/Goal Kicks", "Set Pieces/Throw-Ins",
    "Set Pieces/Freekicks",
}


def video_duration(video):
    return float(subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        text=True,
    ).strip())


def gt_event_timestamps(cfg) -> list[tuple[float, str]]:
    out = []
    for half_idx, fp in enumerate((cfg["gt_h1"], cfg["gt_h2"])):
        d = json.loads(Path(fp).read_text())
        for entry in d.get("data", []):
            game_sec = entry.get("event_time", 0) / 1000.0
            for ev in entry.get("events", []):
                name = ev.get("event_name", "")
                if name not in PREFERRED_EVENTS:
                    continue
                if half_idx == 0:
                    video_sec = game_sec + cfg["video_offset"]
                else:
                    video_sec = (game_sec - cfg["half2_game_offset"]) + cfg["half2_video_start"]
                out.append((video_sec, name))
                break
    return out


def random_play_timestamps(dur: float, n: int, rng: random.Random) -> list[tuple[float, str]]:
    return [(rng.uniform(60, max(60, dur - 60)), "random_play") for _ in range(n)]


def extract_frame(video: Path, ts: float, out_path: Path) -> bool:
    try:
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error",
                        "-ss", f"{ts:.3f}", "-i", str(video),
                        "-frames:v", "1", "-q:v", "2",
                        "-y", str(out_path)],
                       check=True, timeout=30)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def main() -> int:
    rng = random.Random(RANDOM_SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "frames").mkdir(exist_ok=True)
    (OUT_ROOT / "labels").mkdir(exist_ok=True)

    plan: list[dict] = []
    for gid, cfg in GAMES.items():
        video = Path(cfg["video"])
        if not video.exists():
            print(f"  [skip] {gid}: video missing")
            continue
        try:
            dur = video_duration(video)
        except Exception as e:
            print(f"  [skip] {gid}: ffprobe failed: {e}")
            continue

        # GT-anchored frames: shuffle then take first N
        gt = gt_event_timestamps(cfg)
        rng.shuffle(gt)
        gt = gt[:GT_FRAMES_PER_GAME]

        # Random-play frames
        rand = random_play_timestamps(dur, RANDOM_FRAMES_PER_GAME, rng)

        # Combine
        for ts, evname in gt + rand:
            plan.append({"game_id": gid, "video": str(video),
                         "video_ts": ts, "source": evname})
        print(f"  {gid}: {len(gt)} GT + {len(rand)} random = {len(gt)+len(rand)}")

    # Shuffle the whole plan so games are interleaved (less monotonous to annotate)
    rng.shuffle(plan)
    print(f"\ntotal: {len(plan)} frames")

    print("\n[extract] writing frames...")
    t0 = time.time()
    manifest = []
    for i, p in enumerate(plan):
        out_path = OUT_ROOT / "frames" / f"{i:04d}.jpg"
        if extract_frame(Path(p["video"]), p["video_ts"], out_path):
            manifest.append({"idx": i, "game_id": p["game_id"],
                             "video_ts": p["video_ts"], "source": p["source"]})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(plan)}] {time.time()-t0:.0f}s")

    (OUT_ROOT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest)
    )
    print(f"\n=== done in {time.time()-t0:.0f}s ===")
    print(f"  {len(manifest)} frames at {OUT_ROOT}/frames/")
    print(f"  manifest at {OUT_ROOT}/manifest.jsonl")
    print(f"  empty {OUT_ROOT}/labels/ ready for annotation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
