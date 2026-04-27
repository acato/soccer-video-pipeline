#!/usr/bin/env python3
"""QL3 Phase A: build LoRA training/eval dataset from labeled games.

For each (game, calibration) pair in the manifest, this script:
  1. Enumerates inference-shaped windows (step=10s, window=15s) across
     the game's video duration.
  2. For each window, samples 5 frames using the SAME FrameSampler +
     field_crop pipeline that runs at inference (matches train/serve).
  3. Computes the window's GT event labels by intersecting GT-event
     video timestamps with the window range.
  4. Saves frames as JPEGs and writes one jsonl row per window
     containing {prompt, frame_paths, frame_timestamps, target_json,
     game_id, window_idx, window_start, window_end}.

Output layout:
    lora_dataset/
      manifest.json                  # what was actually processed
      frames/game_NN/win_NNN_fN.jpg
      labels/game_NN.jsonl

Usage:
    python prepare_lora_dataset.py \
        --output-dir /Volumes/transit/soccer-finetune/lora_dataset \
        --games game_11 game_01 ...     # subset; default = all calibrated
        [--field-crop true|false]       # default true (matches deployment)

Run-once expectation: ~14 games × ~685 windows × 5 frames = ~48k JPEGs,
~2.4 GB total. ~30-60 min wall-time on a single FrameSampler.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.frame_sampler import FrameSampler


# ──────────────────────────────────────────────────────────────────────────
# GT mapping (mirrors scripts/evaluate_detection.py)
# ──────────────────────────────────────────────────────────────────────────
GT_TO_PIPELINE = {
    "Saves/Catches": "catch",
    "Saves/Parries": "shot_stop_diving",
    "Set Pieces/Corners": "corner_kick",
    "Set Pieces/Goal Kicks": "goal_kick",
    "Set Pieces/Throw-Ins": "throw_in",
    "Set Pieces/Freekicks": "free_kick_shot",
    "Set Pieces/Penalty Kicks": "penalty",
    "Goals Conceded": "goal",
    "Shots & Goals": "shot_on_target",
}


@dataclass
class GameSpec:
    """A single labeled game ready for dataset extraction."""
    game_id: str
    video_path: str
    h1_json: str
    h2_json: str
    video_offset: float       # seconds from video start to 1st-half kickoff
    half2_video_start: float  # seconds from video start to 2nd-half kickoff
    half2_game_offset: float  # GT clock value at 2nd-half kickoff (e.g. 2700 for 45-min halves)
    split: str = "train"      # "train" or "eval"


# ──────────────────────────────────────────────────────────────────────────
# Manifest of all games we have GT for
# ──────────────────────────────────────────────────────────────────────────
NAS = Path("/Volumes/transit/Games")

# All 19 games — calibration recovered from the v4 8B finetune training data
# (/Volumes/transit/soccer-finetune/stage2_all.jsonl _meta fields).
# Per-game offsets back-computed from video_ts vs event_time_ms; zero variance
# across all events in each (game, half), so this is the authoritative source.
# Note: where batch_generalization auto-calibration disagreed with v4 (games
# 1, 4 most notably), v4 wins — past F1 numbers on those games were partially
# noise from calibration drift.
#
# split assignment:
#   eval = the 5 games with established baselines (matches batch_generalization)
#   train = the 14 remaining games
CALIBRATED: list[GameSpec] = [
    # ── Held-out eval (existing benchmarks) ──
    GameSpec(
        game_id="game_01",
        video_path=str(NAS / "1" / "1752784779079_video-3706a2d1-2dc5-4bc0-8dad-c94f0262e36c-1752785855.26862-encoded.mp4"),
        h1_json=str(NAS / "1" / "08 GA (U19) vs Sporting AC_1st Half.json"),
        h2_json=str(NAS / "1" / "08 GA (U19) vs Sporting AC_2nd Half.json"),
        video_offset=320.0, half2_video_start=3808.0, half2_game_offset=2700.0,
        split="eval",
    ),
    GameSpec(
        game_id="game_04",
        video_path=str(NAS / "4" / "1768796597623_seattle-reign-academy-2011-ga-vs-ww-surf-2011-ga-11-ga-vs-ww-surf-jan-18-d74e23a0-0ae6-49de-8ebf-d05cfc700ce9-1768798011.373838-encoded.mp4"),
        h1_json=str(NAS / "4" / "2026-01-18_Seattle Reign 2011 GA (U15) vs NPSA WW Surf U15 (W)_1st Half.json"),
        h2_json=str(NAS / "4" / "2026-01-18_Seattle Reign 2011 GA (U15) vs NPSA WW Surf U15 (W)_2nd Half.json"),
        video_offset=325.0, half2_video_start=3375.0, half2_game_offset=2400.0,
        split="eval",
    ),
    GameSpec(
        game_id="game_10",
        video_path=str(NAS / "10" / "1770526794598_reign-2011-vs-wa-rush-2026-02-07-fddf6486-a04f-40a7-96f6-73ecb7c9b548-1770527705.883494-encoded.mp4"),
        h1_json=str(NAS / "10" / "2026-02-07_Seattle Reign 2011 GA (U15) vs Washington Rush U15 (W)_1st Half.json"),
        h2_json=str(NAS / "10" / "2026-02-07_Seattle Reign 2011 GA (U15) vs Washington Rush U15 (W)_2nd Half.json"),
        video_offset=252.0, half2_video_start=3393.0, half2_game_offset=2400.0,
        split="eval",
    ),
    GameSpec(
        game_id="game_11",  # Rush U19 — primary benchmark
        video_path="/Users/aless/soccer-working/2026-02-07 - Rush - GA2008.mp4",
        h1_json="/Volumes/transit/08 GA (U19) vs Washington Rush U19 (W)_1st Half.json",
        h2_json="/Volumes/transit/08 GA (U19) vs Washington Rush U19 (W)_2nd Half.json",
        video_offset=416.0, half2_video_start=3880.0, half2_game_offset=2700.0,
        split="eval",
    ),
    GameSpec(
        game_id="game_13",
        video_path=str(NAS / "13" / "1772040274070_07_08_GA_at_Capital_FC-47845ea7-c6f8-4118-9c56-cef1866addc8-1772040752.107228-encoded.mp4"),
        h1_json=str(NAS / "13" / "08 GA (U19)_1st Half.json"),
        h2_json=str(NAS / "13" / "08 GA (U19)_2nd Half.json"),
        video_offset=323.0, half2_video_start=3828.0, half2_game_offset=2700.0,
        split="eval",
    ),
    # ── Training set (14 games) ──
    GameSpec(
        game_id="game_02",
        video_path=str(NAS / "2" / "2 event clips merged.mp4"),
        h1_json=str(NAS / "2" / "08 GA (U19)_1st Half.json"),
        h2_json=str(NAS / "2" / "08 GA (U19)_2nd Half.json"),
        video_offset=0.0, half2_video_start=2700.0, half2_game_offset=2700.0,
        split="train",
    ),
    GameSpec(
        game_id="game_03",
        video_path=str(NAS / "3" / "1768376237461_celtic-ga-vs-reign-fc-2025-12-12_1-6de0a0a3-d892-4f6d-8152-32c91d4d74bf-1768377809.843711-encoded.mp4"),
        h1_json=str(NAS / "3" / "2025-12-13_Seattle Celtic U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json"),
        h2_json=str(NAS / "3" / "2025-12-13_Seattle Celtic U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json"),
        video_offset=30.0, half2_video_start=3058.0, half2_game_offset=2400.0,
        split="train",
    ),
    GameSpec(
        game_id="game_05",
        video_path=str(NAS / "5" / "1768795314475_seattle-reign-academy-07-08-ga-vs-wwa-surf-07-08-ga-vs-wwa-surf-bbc88000-2fdc-46ad-a5fa-53387ddcfcde-1768796696.940759-encoded.mp4"),
        h1_json=str(NAS / "5" / "08 GA (U19) vs NPSA WW Surf U19 (W)_1st Half.json"),
        h2_json=str(NAS / "5" / "08 GA (U19) vs NPSA WW Surf U19 (W)_2nd Half.json"),
        video_offset=195.0, half2_video_start=3760.0, half2_game_offset=2700.0,
        split="train",
    ),
    GameSpec(
        game_id="game_06",
        video_path=str(NAS / "6" / "1770844071129_07_08_GA_vs_Spokane_Shadow-136c7e66-a7b7-47a2-8370-c3948156cf14-1770845042.758828-encoded.mp4"),
        h1_json=str(NAS / "6" / "08 GA (U19)_1st Half.json"),
        h2_json=str(NAS / "6" / "08 GA (U19)_2nd Half.json"),
        video_offset=84.0, half2_video_start=3086.0, half2_game_offset=2700.0,
        split="train",
    ),
    GameSpec(
        game_id="game_07",
        video_path=str(NAS / "7" / "1770162674539_recording-578ec162-e5ab-4194-89ca-835f96b4ac85-1770163416.24411-encoded.mp4"),
        h1_json=str(NAS / "7" / "2026-01-31_Spokane Shadow U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json"),
        h2_json=str(NAS / "7" / "2026-01-31_Spokane Shadow U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json"),
        video_offset=114.0, half2_video_start=2760.0, half2_game_offset=2400.0,
        split="train",
    ),
    GameSpec(
        game_id="game_08",
        video_path=str(NAS / "8" / "1770163303827_video-656d16fb-9adb-41c8-a104-0193e5c683d3-1770163984.213979-encoded.mp4"),
        h1_json=str(NAS / "8" / "2026-02-01_Washington East Surf SC U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json"),
        h2_json=str(NAS / "8" / "2026-02-01_Washington East Surf SC U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json"),
        video_offset=615.0, half2_video_start=3712.0, half2_game_offset=2400.0,
        split="train",
    ),
    GameSpec(
        game_id="game_09",
        video_path=str(NAS / "9" / "1770842972307_07_08_GA_vs_WE_Surf-bd754842-cff9-443b-b193-b034ae773ba0-1770843677.884134-encoded.mp4"),
        h1_json=str(NAS / "9" / "08 GA (U19)_1st Half.json"),
        h2_json=str(NAS / "9" / "08 GA (U19)_2nd Half.json"),
        video_offset=768.0, half2_video_start=4262.0, half2_game_offset=2700.0,
        split="train",
    ),
    GameSpec(
        game_id="game_12",
        video_path=str(NAS / "12" / "1771957218570_video_2-1993afe5-f90c-4229-8fba-2bf5773733c3-1771958315.041616-encoded.mp4"),
        h1_json=str(NAS / "12" / "2026-02-21_Capital FC U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json"),
        h2_json=str(NAS / "12" / "2026-02-21_Capital FC U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json"),
        video_offset=1453.0, half2_video_start=4614.0, half2_game_offset=2400.0,
        split="train",
    ),
    GameSpec(
        game_id="game_14",
        video_path=str(NAS / "14" / "1771956414386_video_1-0f3811c9-a14e-4efd-97d9-e4250db90231-1771957281.538088-encoded.mp4"),
        h1_json=str(NAS / "14" / "2026-02-22_Eugene Metro FC U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json"),
        h2_json=str(NAS / "14" / "2026-02-22_Eugene Metro FC U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json"),
        video_offset=87.0, half2_video_start=3194.0, half2_game_offset=2400.0,
        split="train",
    ),
    GameSpec(
        game_id="game_15",
        video_path=str(NAS / "15" / "1772132552044_07_08_GA_at_Eugene_Metros-9efd30af-f4d7-4259-9b0e-1033d7582d15-1772133003.51557-encoded.mp4"),
        h1_json=str(NAS / "15" / "08 GA (U19)_1st Half.json"),
        h2_json=str(NAS / "15" / "08 GA (U19)_2nd Half.json"),
        video_offset=130.0, half2_video_start=3591.0, half2_game_offset=2700.0,
        split="train",
    ),
    GameSpec(
        game_id="game_16",
        video_path=str(NAS / "16" / "1773129740515_2011-ga-vs-cp-2026-03-07-37c9ed14-bcf9-4ed2-a1ff-e08fbb215c61-1773131555.006324-encoded.mp4"),
        h1_json=str(NAS / "16" / "2026-03-07_Seattle Reign 2011 GA (U15) vs Columbia Premier SC U15 (W)_1st Half.json"),
        h2_json=str(NAS / "16" / "2026-03-07_Seattle Reign 2011 GA (U15) vs Columbia Premier SC U15 (W)_2nd Half.json"),
        video_offset=0.0, half2_video_start=2980.0, half2_game_offset=2400.0,
        split="train",
    ),
    GameSpec(
        game_id="game_17",
        video_path=str(NAS / "17" / "1773178741853_0708-ga-vs-columbia-premier-2026-03-07-252c83eb-1e2c-4cfe-ae1d-755422d15f00-1773179894.895267-encoded.mp4"),
        h1_json=str(NAS / "17" / "08 GA (U19) vs Columbia Premier SC U19 (W)_1st Half.json"),
        h2_json=str(NAS / "17" / "08 GA (U19) vs Columbia Premier SC U19 (W)_2nd Half.json"),
        video_offset=56.0, half2_video_start=3608.0, half2_game_offset=2700.0,
        split="train",
    ),
    GameSpec(
        game_id="game_18",
        video_path=str(NAS / "18" / "1773550996606_seattle-reign-academy-2011-ga-vs-westside-metros-2011-ga-seattle-reign-academy-2011-ga-vs-ws-metros-85985259-4639-47c6-899f-76080b93eb9b-1773555300.926346-encoded.mp4"),
        h1_json=str(NAS / "18" / "2026-03-14_Seattle Reign 2011 GA (U15) vs Westside Metros FC U15 (W)_1st Half.json"),
        h2_json=str(NAS / "18" / "2026-03-14_Seattle Reign 2011 GA (U15) vs Westside Metros FC U15 (W)_2nd Half.json"),
        video_offset=78.0, half2_video_start=3223.0, half2_game_offset=2400.0,
        split="train",
    ),
    GameSpec(
        game_id="game_19",
        video_path=str(NAS / "19" / "1773631328983_seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-1aa48cf7-b31f-41c9-aa17-cef4b4c22267-1773632512.672041-encoded.mp4"),
        h1_json=str(NAS / "19" / "2026-03-15_Seattle Reign 2011 GA (U15) vs Oregon Premier FC U15 (W)_1st Half.json"),
        h2_json=str(NAS / "19" / "2026-03-15_Seattle Reign 2011 GA (U15) vs Oregon Premier FC U15 (W)_2nd Half.json"),
        video_offset=0.0, half2_video_start=2863.0, half2_game_offset=2400.0,
        split="train",
    ),
]

# All 19 games calibrated above. Legacy placeholder retained as []
# so existing manifest emit logic doesn't break.
UNCALIBRATED: list[dict] = []
_LEGACY_UNCALIBRATED_REFERENCE: list[dict] = [
    {"game_id": "game_02", "subdir": "2",
     "video_filename": "2 event clips merged.mp4",
     "h1_filename": "08 GA (U19)_1st Half.json",
     "h2_filename": "08 GA (U19)_2nd Half.json",
     "half2_game_offset": 2700.0},  # U19
    {"game_id": "game_03", "subdir": "3",
     "video_filename": "1768376237461_celtic-ga-vs-reign-fc-2025-12-12_1-6de0a0a3-d892-4f6d-8152-32c91d4d74bf-1768377809.843711-encoded.mp4",
     "h1_filename": "2025-12-13_Seattle Celtic U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json",
     "h2_filename": "2025-12-13_Seattle Celtic U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json",
     "half2_game_offset": 2400.0},  # U15
    {"game_id": "game_05", "subdir": "5",
     "video_filename": "1768795314475_seattle-reign-academy-07-08-ga-vs-wwa-surf-07-08-ga-vs-wwa-surf-bbc88000-2fdc-46ad-a5fa-53387ddcfcde-1768796696.940759-encoded.mp4",
     "h1_filename": "08 GA (U19) vs NPSA WW Surf U19 (W)_1st Half.json",
     "h2_filename": "08 GA (U19) vs NPSA WW Surf U19 (W)_2nd Half.json",
     "half2_game_offset": 2700.0},
    {"game_id": "game_06", "subdir": "6",
     "video_filename": "1770844071129_07_08_GA_vs_Spokane_Shadow-136c7e66-a7b7-47a2-8370-c3948156cf14-1770845042.758828-encoded.mp4",
     "h1_filename": "08 GA (U19)_1st Half.json",
     "h2_filename": "08 GA (U19)_2nd Half.json",
     "half2_game_offset": 2700.0},
    {"game_id": "game_07", "subdir": "7",
     "video_filename": "1770162674539_recording-578ec162-e5ab-4194-89ca-835f96b4ac85-1770163416.24411-encoded.mp4",
     "h1_filename": "2026-01-31_Spokane Shadow U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json",
     "h2_filename": "2026-01-31_Spokane Shadow U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json",
     "half2_game_offset": 2400.0},
    {"game_id": "game_08", "subdir": "8",
     "video_filename": "1770163303827_video-656d16fb-9adb-41c8-a104-0193e5c683d3-1770163984.213979-encoded.mp4",
     "h1_filename": "2026-02-01_Washington East Surf SC U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json",
     "h2_filename": "2026-02-01_Washington East Surf SC U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json",
     "half2_game_offset": 2400.0},
    {"game_id": "game_09", "subdir": "9",
     "video_filename": "1770842972307_07_08_GA_vs_WE_Surf-bd754842-cff9-443b-b193-b034ae773ba0-1770843677.884134-encoded.mp4",
     "h1_filename": "08 GA (U19)_1st Half.json",
     "h2_filename": "08 GA (U19)_2nd Half.json",
     "half2_game_offset": 2700.0},
    {"game_id": "game_12", "subdir": "12",
     "video_filename": "1771957218570_video_2-1993afe5-f90c-4229-8fba-2bf5773733c3-1771958315.041616-encoded.mp4",
     "h1_filename": "2026-02-21_Capital FC U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json",
     "h2_filename": "2026-02-21_Capital FC U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json",
     "half2_game_offset": 2400.0},
    {"game_id": "game_14", "subdir": "14",
     "video_filename": "1771956414386_video_1-0f3811c9-a14e-4efd-97d9-e4250db90231-1771957281.538088-encoded.mp4",
     "h1_filename": "2026-02-22_Eugene Metro FC U15 (W) vs Seattle Reign 2011 GA (U15)_1st Half.json",
     "h2_filename": "2026-02-22_Eugene Metro FC U15 (W) vs Seattle Reign 2011 GA (U15)_2nd Half.json",
     "half2_game_offset": 2400.0},
    {"game_id": "game_15", "subdir": "15",
     "video_filename": "1772132552044_07_08_GA_at_Eugene_Metros-9efd30af-f4d7-4259-9b0e-1033d7582d15-1772133003.51557-encoded.mp4",
     "h1_filename": "08 GA (U19)_1st Half.json",
     "h2_filename": "08 GA (U19)_2nd Half.json",
     "half2_game_offset": 2700.0},
    {"game_id": "game_16", "subdir": "16",
     "video_filename": "1773129740515_2011-ga-vs-cp-2026-03-07-37c9ed14-bcf9-4ed2-a1ff-e08fbb215c61-1773131555.006324-encoded.mp4",
     "h1_filename": "2026-03-07_Seattle Reign 2011 GA (U15) vs Columbia Premier SC U15 (W)_1st Half.json",
     "h2_filename": "2026-03-07_Seattle Reign 2011 GA (U15) vs Columbia Premier SC U15 (W)_2nd Half.json",
     "half2_game_offset": 2400.0},
    {"game_id": "game_17", "subdir": "17",
     "video_filename": "1773178741853_0708-ga-vs-columbia-premier-2026-03-07-252c83eb-1e2c-4cfe-ae1d-755422d15f00-1773179894.895267-encoded.mp4",
     "h1_filename": "08 GA (U19) vs Columbia Premier SC U19 (W)_1st Half.json",
     "h2_filename": "08 GA (U19) vs Columbia Premier SC U19 (W)_2nd Half.json",
     "half2_game_offset": 2700.0},
    {"game_id": "game_18", "subdir": "18",
     "video_filename": "1773550996606_seattle-reign-academy-2011-ga-vs-westside-metros-2011-ga-seattle-reign-academy-2011-ga-vs-ws-metros-85985259-4639-47c6-899f-76080b93eb9b-1773555300.926346-encoded.mp4",
     "h1_filename": "2026-03-14_Seattle Reign 2011 GA (U15) vs Westside Metros FC U15 (W)_1st Half.json",
     "h2_filename": "2026-03-14_Seattle Reign 2011 GA (U15) vs Westside Metros FC U15 (W)_2nd Half.json",
     "half2_game_offset": 2400.0},
    {"game_id": "game_19", "subdir": "19",
     "video_filename": "1773631328983_seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-1aa48cf7-b31f-41c9-aa17-cef4b4c22267-1773632512.672041-encoded.mp4",
     "h1_filename": "2026-03-15_Seattle Reign 2011 GA (U15) vs Oregon Premier FC U15 (W)_1st Half.json",
     "h2_filename": "2026-03-15_Seattle Reign 2011 GA (U15) vs Oregon Premier FC U15 (W)_2nd Half.json",
     "half2_game_offset": 2400.0},
]


def load_gt_events(spec: GameSpec) -> list[dict]:
    """Load GT events as a list of {type, video_sec, gt_name, team}.

    Mirrors evaluate_detection.load_ground_truth but returns dicts instead
    of GTEvent dataclasses for easier serialization.
    """
    out = []
    for half_idx, json_path in enumerate([spec.h1_json, spec.h2_json]):
        with open(json_path) as f:
            data = json.load(f)
        for entry in data["data"]:
            ms = entry["event_time"]
            for ev in entry.get("events", []):
                name = ev["event_name"]
                prop = ev.get("property", {})
                typ = prop.get("Type", "")
                key = name + ("/" + typ if typ else "")
                ptype = GT_TO_PIPELINE.get(key)
                if ptype is None:
                    continue
                game_sec = ms / 1000.0
                if half_idx == 0:
                    video_sec = game_sec + spec.video_offset
                else:
                    video_sec = (game_sec - spec.half2_game_offset) + spec.half2_video_start
                out.append({
                    "type": ptype,
                    "video_sec": video_sec,
                    "game_sec": game_sec,
                    "half": half_idx,
                    "gt_name": key,
                    "team": entry.get("team_name", ""),
                })
    return out


def video_duration(video_path: str) -> float:
    import subprocess
    p = subprocess.run(
        ["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
         video_path],
        capture_output=True, text=True, check=True,
    )
    return float(p.stdout.strip())


def compute_field_bbox(sampler: FrameSampler, duration: float,
                        n_samples: int = 20) -> Optional[tuple]:
    """Replicate DualPassDetector._compute_field_bbox without instantiating
    a full detector. Returns (x1, y1, x2, y2) normalized or None."""
    import numpy as np
    from PIL import Image
    ts_list = [duration * (0.01 + 0.98 * (i + 0.5) / n_samples) for i in range(n_samples)]
    agg = None
    n = 0
    for t in ts_list:
        try:
            jpeg = sampler._extract_single_frame(t)
        except Exception:
            continue
        if not jpeg:
            continue
        try:
            img = Image.open(io.BytesIO(jpeg)).convert("HSV")
            arr = np.array(img)
        except Exception:
            continue
        h = arr[..., 0]; s = arr[..., 1]; v = arr[..., 2]
        mask = ((h >= 40) & (h <= 130) & (s >= 50) & (v >= 40)).astype(np.float32)
        if agg is None:
            agg = mask
        elif mask.shape == agg.shape:
            agg = agg + mask
        else:
            continue
        n += 1
    if agg is None or n == 0:
        return None
    agg = agg / float(n)
    H, W = agg.shape
    row_density = agg.mean(axis=1)
    col_density = agg.mean(axis=0)
    row_mask = row_density >= 0.30
    col_mask = col_density >= 0.30
    if not row_mask.any() or not col_mask.any():
        return None
    y1 = int(np.argmax(row_mask))
    y2 = int(H - np.argmax(row_mask[::-1]))
    x1 = int(np.argmax(col_mask))
    x2 = int(W - np.argmax(col_mask[::-1]))
    pad_x = int(0.03 * W); pad_y = int(0.03 * H)
    y1 = max(0, y1 - pad_y); y2 = min(H, y2 + pad_y)
    x1 = max(0, x1 - pad_x); x2 = min(W, x2 + pad_x)
    return (x1 / W, y1 / H, x2 / W, y2 / H)


def apply_field_crop(jpeg_bytes: bytes, bbox: Optional[tuple]) -> bytes:
    """Crop a JPEG to the cached field bbox (matches dual_pass_detector behavior)."""
    if bbox is None:
        return jpeg_bytes
    x1, y1, x2, y2 = bbox
    if (x2 - x1) * (y2 - y1) >= 0.98:
        return jpeg_bytes
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg_bytes))
    W, H = img.size
    cropped = img.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def gt_events_in_window(gt_events: list[dict], win_start: float,
                        win_end: float) -> list[dict]:
    return [g for g in gt_events if win_start <= g["video_sec"] <= win_end]


def label_for_window(events_in_win: list[dict], win_start: float,
                      win_end: float) -> list[dict]:
    """Build the target JSON list the model should emit for this window."""
    if not events_in_win:
        return [{
            "event_type": "none",
            "start_sec": win_start,
            "end_sec": win_end,
            "confidence": 0.9,
            "reasoning": "no GT event in window",
        }]
    out = []
    for g in events_in_win:
        # Use a tight 1-second start/end around the GT timestamp; reasoning
        # field gets a synthetic placeholder ("derived from GT").
        out.append({
            "event_type": g["type"],
            "start_sec": round(g["video_sec"] - 0.5, 1),
            "end_sec": round(g["video_sec"] + 0.5, 1),
            "confidence": 1.0,
            "reasoning": f"GT label ({g['gt_name']})",
        })
    return out


def process_game(spec: GameSpec, output_dir: Path, *,
                  field_crop_enabled: bool = True,
                  step_sec: float = 10.0,
                  window_sec: float = 15.0,
                  n_frames: int = 5,
                  frame_width: int = 1280,
                  max_windows: Optional[int] = None) -> dict:
    """Build per-window training rows for one game."""
    print(f"\n=== {spec.game_id}  ({spec.split}) ===")
    print(f"  video: {spec.video_path}")
    duration = video_duration(spec.video_path)
    print(f"  duration: {duration:.0f}s ({duration/60:.1f}m)")

    gt_events = load_gt_events(spec)
    from collections import Counter
    gt_type_counts = Counter(g["type"] for g in gt_events)
    print(f"  GT events: {len(gt_events)}  by type: {dict(gt_type_counts)}")

    # FrameSampler matches inference (frame_width=1280)
    sampler = FrameSampler(spec.video_path, frame_width=frame_width)

    bbox = None
    if field_crop_enabled:
        print(f"  computing field_crop bbox over 20 sample frames...")
        bbox = compute_field_bbox(sampler, duration)
        if bbox is None:
            print(f"  WARNING: field_crop bbox failed, using full frame")
        else:
            print(f"  field_crop bbox: x=[{bbox[0]:.3f},{bbox[2]:.3f}]  "
                  f"y=[{bbox[1]:.3f},{bbox[3]:.3f}]  "
                  f"area_frac={(bbox[2]-bbox[0])*(bbox[3]-bbox[1]):.3f}")

    frames_dir = output_dir / "frames" / spec.game_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / f"{spec.game_id}.jsonl"

    n_windows_total = int(duration / step_sec) + 1
    n_processed = 0
    n_with_events = 0
    type_in_label_counter = Counter()

    with open(labels_path, "w") as label_fh:
        t = 0.0
        win_idx = 0
        while t < duration:
            if max_windows and win_idx >= max_windows:
                break
            win_start = max(0.0, t - (window_sec - step_sec) / 2)
            win_end = min(duration, win_start + window_sec)
            center = (win_start + win_end) / 2
            half_span = (win_end - win_start) / 2
            interval = max(1.0, (win_end - win_start) / n_frames)

            frames = sampler.sample_range(
                center_sec=center, window_sec=half_span,
                interval_sec=interval, duration_sec=duration,
            )
            if not frames:
                t += step_sec; win_idx += 1; continue
            if len(frames) > n_frames:
                s = len(frames) / n_frames
                idxs = [int(i * s) for i in range(n_frames)]
                frames = [frames[i] for i in idxs]

            # Save frames (with field_crop applied) to disk
            frame_paths = []
            frame_timestamps = []
            for fi, fr in enumerate(frames):
                jpeg = apply_field_crop(fr.jpeg_bytes, bbox) if bbox else fr.jpeg_bytes
                fp = frames_dir / f"win_{win_idx:04d}_f{fi}.jpg"
                fp.write_bytes(jpeg)
                frame_paths.append(str(fp.relative_to(output_dir)))
                frame_timestamps.append(round(float(fr.timestamp_sec), 2))

            # Label
            in_win = gt_events_in_window(gt_events, win_start, win_end)
            target = label_for_window(in_win, win_start, win_end)
            for ev in target:
                type_in_label_counter[ev["event_type"]] += 1

            row = {
                "game_id": spec.game_id,
                "split": spec.split,
                "window_idx": win_idx,
                "window_start_sec": round(win_start, 2),
                "window_end_sec": round(win_end, 2),
                "frames": frame_paths,
                "frame_timestamps": frame_timestamps,
                "n_gt_events_in_window": len(in_win),
                "target": target,
            }
            label_fh.write(json.dumps(row) + "\n")

            n_processed += 1
            if in_win:
                n_with_events += 1

            if win_idx % 100 == 0:
                print(f"  win {win_idx}/{n_windows_total}  "
                      f"events_so_far={n_with_events}", end="\r")
            t += step_sec; win_idx += 1

    print(f"  done: {n_processed} windows ({n_with_events} with events, "
          f"{100*n_with_events/n_processed:.1f}%)")
    print(f"  target type distribution: {dict(type_in_label_counter)}")

    return {
        "game_id": spec.game_id,
        "split": spec.split,
        "duration_sec": duration,
        "n_windows": n_processed,
        "n_windows_with_events": n_with_events,
        "n_gt_events": len(gt_events),
        "gt_type_counts": dict(gt_type_counts),
        "target_type_counts": dict(type_in_label_counter),
        "field_bbox": bbox,
        "labels_path": str(labels_path.relative_to(output_dir)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--games", nargs="*", default=None,
                    help="Game IDs to process (default: all calibrated)")
    ap.add_argument("--no-field-crop", action="store_true",
                    help="Skip field_crop (default: apply, matching deployment)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="Cap windows per game (for smoke testing)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_id = {g.game_id: g for g in CALIBRATED}
    if args.games:
        unknown = [g for g in args.games if g not in by_id]
        if unknown:
            print(f"WARNING: unknown game IDs (not in CALIBRATED list): {unknown}",
                  file=sys.stderr)
        targets = [by_id[g] for g in args.games if g in by_id]
    else:
        targets = list(CALIBRATED)

    summaries = []
    for spec in targets:
        try:
            s = process_game(spec, output_dir,
                              field_crop_enabled=not args.no_field_crop,
                              max_windows=args.max_windows)
            summaries.append(s)
        except Exception as exc:
            print(f"\n  FAILED on {spec.game_id}: {exc}", file=sys.stderr)
            summaries.append({"game_id": spec.game_id, "error": str(exc)})

    train_games = [s["game_id"] for s in summaries
                   if "error" not in s and s.get("split") == "train"]
    eval_games = [s["game_id"] for s in summaries
                  if "error" not in s and s.get("split") == "eval"]
    manifest = {
        "games_processed": [s for s in summaries if "error" not in s],
        "failures": [s for s in summaries if "error" in s],
        "train_games": train_games,
        "eval_games": eval_games,
        "field_crop_applied": not args.no_field_crop,
        "calibration_source": "v4 8B finetune training data (stage2_all.jsonl _meta)",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDataset prep done. Manifest: {output_dir / 'manifest.json'}")
    total_w = sum(s.get("n_windows", 0) for s in summaries if "error" not in s)
    total_e = sum(s.get("n_windows_with_events", 0) for s in summaries if "error" not in s)
    print(f"Totals: {len(summaries)} games, {total_w} windows, {total_e} with events")


if __name__ == "__main__":
    main()
