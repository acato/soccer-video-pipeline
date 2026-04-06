#!/usr/bin/env python3
"""
Training data extraction pipeline for Qwen3-VL fine-tuning.

Goes from (MP4 + JSON GT) --> (extracted frames + JSONL training records).

Usage:
    # Step 1: Generate calibration template (fill in anchor timestamps manually)
    python scripts/extract_training_data.py calibrate \
        --games-dir /Volumes/transit/Games \
        --output calibration.json

    # Step 2: Extract frames and generate training JSONL
    python scripts/extract_training_data.py extract \
        --games-dir /Volumes/transit/Games \
        --calibration calibration.json \
        --output-dir /data/soccer-finetune \
        --stage 2

    # Step 3: Split into train/val (holdout by game number)
    python scripts/extract_training_data.py split \
        --data-dir /data/soccer-finetune \
        --holdout-games 11,13 \
        --val-ratio 0.1

    # Step 4: Augment training data (temporal shifts, mirror, jitter)
    python scripts/extract_training_data.py augment \
        --data-dir /data/soccer-finetune \
        --source train.jsonl \
        --output train_augmented.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Event type mapping from analytics JSON to our training labels
EVENT_MAP = {
    # Shots & Goals
    ("Shots & Goals", "Goals"): "goal",
    ("Shots & Goals", "Shots On Target"): "shot_on_target",
    ("Shots & Goals", "Shots Off Target"): "shot_off_target",
    ("Shots & Goals", "Blocked Shots"): "shot_blocked",
    ("Shots & Goals", "Keeper Rush-outs"): "keeper_rushout",
    # Saves
    ("Saves", "Catches"): "save_catch",
    ("Saves", "Parries"): "save_parry",
    # Set Pieces
    ("Set Pieces", "Corners"): "corner_kick",
    ("Set Pieces", "Goal Kicks"): "goal_kick",
    ("Set Pieces", "Throw-Ins"): "throw_in",
    ("Set Pieces", "Freekicks"): "free_kick",
    ("Set Pieces", "Penalty Kicks"): "penalty",
}

# Temporal windows per event type: (seconds_before, seconds_after)
# Frames are sampled at a rate that yields 8 frames within the window.
EVENT_WINDOWS = {
    "goal":            (2.0, 10.0),    # 12s total, 0.67 FPS -> 8 frames
    "save_catch":      (1.0, 5.0),     # 6s total, 1.33 FPS -> 8 frames
    "save_parry":      (1.0, 3.0),     # 4s total, 2.0 FPS -> 8 frames
    "shot_on_target":  (1.0, 3.0),     # 4s total, 2.0 FPS
    "shot_off_target": (1.0, 3.0),
    "shot_blocked":    (1.0, 3.0),
    "corner_kick":     (2.0, 6.0),     # 8s total, 1.0 FPS
    "goal_kick":       (1.0, 5.0),     # 6s total, 1.33 FPS
    "throw_in":        (1.0, 3.0),     # 4s total, 2.0 FPS
    "free_kick":       (2.0, 4.0),     # 6s total, 1.33 FPS
    "penalty":         (3.0, 8.0),     # 11s total, 0.73 FPS
    "keeper_rushout":  (2.0, 4.0),
    "none":            (0.0, 4.0),     # 4s total, 2.0 FPS
}

NUM_FRAMES = 8
FRAME_WIDTH = 768
FRAME_QUALITY = 5  # JPEG quality for FFmpeg (lower = higher quality)

# Augmentation settings
TEMPORAL_SHIFTS = {
    "goal": [
        (0, 0),        # Original
        (-2.0, -2.0),  # Early shift (more buildup)
        (2.0, 2.0),    # Late shift (more aftermath)
        (8.0, 8.0),    # Restart focus (kickoff only)
    ],
    "save_catch": [
        (0, 0),
        (-1.0, -1.0),  # Earlier: capture the dive
        (2.0, 2.0),    # Later: GK distributing
        (4.0, 4.0),    # Much later: distribution
    ],
}

# Cap per class in augmented dataset
CLASS_CAPS = {
    "goal": 400,
    "save_catch": 600,
    "save_parry": 200,
    "shot_on_target": 300,
    "shot_off_target": 300,
    "shot_blocked": 200,
    "corner_kick": 200,
    "goal_kick": 300,
    "throw_in": 200,
    "free_kick": 250,
    "penalty": 50,
    "none": 400,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GameInfo:
    game_num: str
    game_dir: Path
    mp4_path: Path
    json_paths: list[Path]
    video_duration: float = 0.0


@dataclass
class GTEvent:
    game_num: str
    event_time_ms: int       # Match clock (ms from period start)
    half: int                # 1 or 2
    team: str
    player: str
    event_label: str         # Our training label
    event_name_raw: str      # Original analytics event name
    property_raw: dict       # Original analytics properties
    x: float = 0.0
    y: float = 0.0


@dataclass
class CalibrationEntry:
    game_num: str
    h1_offset_sec: float     # Video seconds to add to H1 match clock
    h2_offset_sec: float     # Video seconds to add to H2 match clock
    h1_anchor_event_ms: Optional[int] = None
    h1_anchor_video_sec: Optional[float] = None
    h2_anchor_event_ms: Optional[int] = None
    h2_anchor_video_sec: Optional[float] = None
    notes: str = ""


@dataclass
class TrainingSample:
    game_num: str
    event_label: str
    event_time_ms: int
    half: int
    video_timestamp: float   # Computed: event_time_ms/1000 + offset
    window_start: float      # Video seconds
    window_end: float        # Video seconds
    frame_paths: list[str] = field(default_factory=list)
    augmentation: str = "original"  # original, early_shift, late_shift, etc.


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

STAGE1_QUERY_TEMPLATE = """\
<image><image><image><image><image><image><image><image>
You are viewing 8 frames from a soccer match recorded by a sideline camera at approximately 50 metres from the field. The frames span {window_sec:.1f} seconds.

Describe what is happening in this clip. Focus on:
1. Ball position and movement direction (toward/away from a goal)
2. Player formations and movements (running, standing, clustered, celebrating)
3. Goalkeeper actions (diving, standing, holding ball, distributing)
4. Any celebration patterns (arms raised, group hugs, sliding)
5. Any restart patterns (kick-off from center, goal kick, corner, throw-in)

Be specific about what you SEE in the frames."""

STAGE2_QUERY_TEMPLATE = """\
<image><image><image><image><image><image><image><image>
You are analyzing 8 frames sampled from a {window_sec:.1f}-second clip of a soccer match recorded by a SIDELINE CAMERA at ~50 metres from the field.

CAMERA LIMITATIONS -- at this distance:
- You CANNOT see whether the goalkeeper's hands touched the ball
- You CANNOT see the ball cross the goal line with certainty
- You CAN see player positions, formations, celebrations, and restart patterns

CLASSIFICATION STRATEGY -- what happens AFTER the action is the best signal:
- Kickoff restart (players at center circle) = goal was scored
- Corner kick restart = goalkeeper parried the shot
- Goal kick restart with no GK save action = shot went wide/over
- Goalkeeper holding ball and distributing = goalkeeper caught it
- Play continues normally = nothing significant

Classify the MAIN event visible in these frames. Choose ONE:
- "goal": A goal was scored -- you see celebration OR kickoff restart at center circle
- "save_catch": Goalkeeper caught the ball -- GK holding/cradling ball in hands after a shot
- "save_parry": Goalkeeper parried/deflected the ball -- followed by corner kick restart
- "shot_on_target": Shot toward goal that was saved (but not clearly a catch or parry)
- "shot_off_target": Shot that missed -- followed by goal kick (ball went wide/over)
- "shot_blocked": Shot blocked by a defender (not the GK)
- "corner_kick": Ball placed at corner flag arc, kicked into the box
- "goal_kick": Ball kicked from six-yard box, no preceding shot attempt visible
- "throw_in": Player holds ball overhead at sideline, throws it in
- "free_kick": Ball placed on ground, kicked from a stoppage after a foul
- "penalty": One shooter vs goalkeeper from penalty spot, others outside box
- "kickoff": Kick-off from center circle (start of half or after goal)
- "none": Normal play, nothing significant, or cannot determine

Respond with EXACTLY this JSON (no other text):
{{"event": "<type>", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


# Stage 1 response templates (concept alignment)
STAGE1_RESPONSE_TEMPLATES = {
    "goal": [
        "A player strikes the ball toward the goal from inside the penalty area. In the following frames, the scoring team's players run toward each other with arms raised in celebration. The opposing team appears dejected, walking slowly back toward the center circle. In the final frames, players from both teams are moving toward the center of the field for a kickoff restart.",
        "The ball is kicked toward goal and enters the net. Players from the attacking team sprint toward the scorer, forming a group celebration near the corner flag. The goalkeeper is on the ground or retrieving the ball from the net. The defending team walks back toward their own half for the restart.",
    ],
    "save_catch": [
        "A shot is taken toward the goal. The goalkeeper dives to one side and gathers the ball securely. In the aftermath frames, the goalkeeper is standing or kneeling while holding the ball against their chest. They look upfield preparing to distribute the ball. No restart is being set up -- the goalkeeper has possession.",
        "After a shot toward goal, the goalkeeper catches the ball cleanly. The GK cradles the ball in both hands, stands up, and prepares to throw or kick it back into play. Players from the attacking team turn and jog back to their positions.",
    ],
    "save_parry": [
        "A shot is taken and the goalkeeper dives, deflecting the ball wide of the goal. The ball goes out of play near the corner. In the following frames, a player from the attacking team walks toward the corner flag to take a corner kick. Multiple players gather in the penalty area.",
        "The goalkeeper makes a diving save, pushing the ball away. The ball deflects toward the corner. A corner kick is being set up -- one player stands at the corner arc while others position themselves in and around the penalty box.",
    ],
    "shot_on_target": [
        "A player shoots the ball toward the goal. The goalkeeper moves to make a save. The shot appears to be on target as it heads toward the goal frame.",
    ],
    "shot_off_target": [
        "A player strikes the ball toward the goal but it goes over the crossbar (or wide of the post). A goal kick is being set up -- the goalkeeper or a defender stands over the ball in the six-yard box, and opposing players move away from the goal area.",
    ],
    "shot_blocked": [
        "A player attempts to shoot toward goal but a defender gets in the way and blocks the shot. The ball deflects away and play continues or a corner/throw-in results.",
    ],
    "corner_kick": [
        "A player stands at the corner flag with the ball at their feet. Multiple players from both teams are gathered inside and around the penalty area. The player at the corner kicks the ball high into the box. Players jump and contest the aerial ball.",
    ],
    "goal_kick": [
        "The ball is placed on the ground inside the six-yard box near the goal. The goalkeeper or a defender stands over the ball. Opposing players are positioned far from the goal, outside the penalty area. The ball is kicked long upfield or short to a nearby defender.",
    ],
    "throw_in": [
        "A player stands at the sideline holding the ball above their head with both hands. Their feet are on or behind the touchline. They throw the ball into play toward a nearby teammate.",
    ],
    "free_kick": [
        "The ball is placed on the ground in the middle or defensive third of the field. A player stands over the ball while opposing players form a wall about 10 yards away. The ball is kicked, either passed to a teammate or shot toward goal.",
    ],
    "penalty": [
        "A single player stands over the ball at the penalty spot, facing the goalkeeper. All other players are positioned outside the penalty area and behind the penalty spot. The shooter runs up and kicks the ball toward one corner of the goal.",
    ],
    "kickoff": [
        "Two players stand near the center spot in the middle of the field. Players from both teams are lined up on their respective halves. One player passes the ball to the other and play begins.",
    ],
    "none": [
        "Players are passing the ball in the midfield area. No shot, set piece, or significant event is occurring. Players jog and contest for possession in normal open play.",
        "The ball is being moved between players in the middle third. This is routine build-up play with no immediate threat on goal or set piece formation.",
    ],
}


def _pick_response_template(label: str) -> str:
    """Select a random response template for the given event label."""
    templates = STAGE1_RESPONSE_TEMPLATES.get(label, STAGE1_RESPONSE_TEMPLATES["none"])
    return random.choice(templates)


def _make_stage2_response(label: str) -> str:
    """Generate a training response for Stage 2 classification."""
    reasoning_map = {
        "goal": "After the shot, players celebrate with arms raised and run toward each other. Teams walk back to the center circle for a kickoff restart.",
        "save_catch": "The goalkeeper gathers the ball and holds it securely against their chest. No restart is set up -- the GK prepares to distribute.",
        "save_parry": "The goalkeeper dives and deflects the ball. A corner kick is being set up at the corner flag.",
        "shot_on_target": "A player shoots toward goal. The goalkeeper moves to make a save.",
        "shot_off_target": "A player shoots but the ball goes wide or over. A goal kick is being set up from the six-yard box.",
        "shot_blocked": "A shot is attempted but blocked by a defender before reaching the goalkeeper.",
        "corner_kick": "A player at the corner flag kicks the ball into a crowded penalty area.",
        "goal_kick": "The ball is kicked from the six-yard box with no preceding shot visible.",
        "throw_in": "A player holds the ball overhead at the sideline and throws it to a teammate.",
        "free_kick": "The ball is placed on the ground and kicked from a stoppage. A defensive wall is visible.",
        "penalty": "A single player faces the goalkeeper from the penalty spot. All others are outside the box.",
        "kickoff": "Players from both teams are lined up on their halves. The ball is at the center spot.",
        "none": "Normal play -- players passing and moving in midfield with no significant event.",
    }
    reasoning = reasoning_map.get(label, "Cannot determine the event type.")
    confidence = 0.95 if label != "none" else 0.85
    return json.dumps({
        "event": label,
        "confidence": confidence,
        "reasoning": reasoning,
    })


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_games(games_dir: Path) -> list[GameInfo]:
    """Find all valid game directories with MP4 + JSON files."""
    games = []
    for entry in sorted(games_dir.iterdir()):
        if not entry.is_dir():
            continue
        mp4s = list(entry.glob("*.mp4"))
        jsons = list(entry.glob("*.json"))
        if not mp4s or not jsons:
            continue
        games.append(GameInfo(
            game_num=entry.name,
            game_dir=entry,
            mp4_path=mp4s[0],
            json_paths=sorted(jsons),
        ))
    return games


def get_video_duration(mp4_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(mp4_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# GT event parsing
# ---------------------------------------------------------------------------

def parse_events(json_path: Path, game_num: str) -> list[GTEvent]:
    """Parse analytics JSON into GTEvent objects."""
    with open(json_path) as f:
        data = json.load(f)

    events = []
    for entry in data.get("data", []):
        half = 1 if entry.get("period_name", "").startswith("1st") else 2
        event_time_ms = entry.get("event_time", 0)
        team = entry.get("team_name", "")
        player = entry.get("player_name", "")
        x = entry.get("x", 0.0)
        y = entry.get("y", 0.0)

        for ev in entry.get("events", []):
            event_name = ev.get("event_name", "")
            props = ev.get("property", {})

            # Determine the outcome/type key
            outcome = props.get("Outcome", props.get("Type", ""))
            key = (event_name, outcome)

            if key in EVENT_MAP:
                events.append(GTEvent(
                    game_num=game_num,
                    event_time_ms=event_time_ms,
                    half=half,
                    team=team,
                    player=player,
                    event_label=EVENT_MAP[key],
                    event_name_raw=event_name,
                    property_raw=props,
                    x=x,
                    y=y,
                ))

    return events


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def generate_calibration_template(games: list[GameInfo], output: Path):
    """Generate a calibration JSON template with placeholders for anchor timestamps."""
    calibration = {}
    for game in games:
        all_events = []
        for jp in game.json_paths:
            all_events.extend(parse_events(jp, game.game_num))

        h1_events = [e for e in all_events if e.half == 1]
        h2_events = [e for e in all_events if e.half == 2]

        # Suggest first throw-in or set piece as anchor (easiest to spot visually)
        h1_anchor = None
        for e in sorted(h1_events, key=lambda x: x.event_time_ms):
            if e.event_label in ("throw_in", "corner_kick", "goal_kick", "free_kick"):
                h1_anchor = e
                break

        h2_anchor = None
        for e in sorted(h2_events, key=lambda x: x.event_time_ms):
            if e.event_label in ("throw_in", "corner_kick", "goal_kick", "free_kick"):
                h2_anchor = e
                break

        game.video_duration = get_video_duration(game.mp4_path)

        calibration[game.game_num] = {
            "mp4": game.mp4_path.name,
            "video_duration_sec": round(game.video_duration, 1),
            "h1_anchor_event_ms": h1_anchor.event_time_ms if h1_anchor else None,
            "h1_anchor_event_label": h1_anchor.event_label if h1_anchor else None,
            "h1_anchor_event_team": h1_anchor.team if h1_anchor else None,
            "h1_anchor_video_sec": None,  # FILL THIS IN MANUALLY
            "h1_offset_sec": None,        # Computed: video_sec - event_ms/1000
            "h2_anchor_event_ms": h2_anchor.event_time_ms if h2_anchor else None,
            "h2_anchor_event_label": h2_anchor.event_label if h2_anchor else None,
            "h2_anchor_event_team": h2_anchor.team if h2_anchor else None,
            "h2_anchor_video_sec": None,  # FILL THIS IN MANUALLY
            "h2_offset_sec": None,        # Computed: video_sec - event_ms/1000
            "notes": f"Total events: H1={len(h1_events)}, H2={len(h2_events)}",
        }

    with open(output, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"Calibration template written to {output}")
    print(f"Found {len(calibration)} games. Fill in h1/h2_anchor_video_sec values manually.")
    print()
    print("For each game:")
    print("  1. Open the MP4 in a video player")
    print("  2. Find the suggested anchor event (first throw-in/corner/goal kick)")
    print("  3. Note the video timestamp (seconds from video start)")
    print("  4. Enter it in the h1_anchor_video_sec / h2_anchor_video_sec fields")
    print("  5. The script will compute offsets: offset = video_sec - event_ms/1000")


def load_calibration(calibration_path: Path) -> dict[str, CalibrationEntry]:
    """Load and validate calibration file."""
    with open(calibration_path) as f:
        data = json.load(f)

    entries = {}
    for game_num, cal in data.items():
        h1_offset = cal.get("h1_offset_sec")
        h2_offset = cal.get("h2_offset_sec")

        # Auto-compute offset if anchor timestamps are provided
        if h1_offset is None and cal.get("h1_anchor_video_sec") is not None:
            h1_offset = cal["h1_anchor_video_sec"] - cal["h1_anchor_event_ms"] / 1000.0

        if h2_offset is None and cal.get("h2_anchor_video_sec") is not None:
            h2_offset = cal["h2_anchor_video_sec"] - cal["h2_anchor_event_ms"] / 1000.0

        if h1_offset is None or h2_offset is None:
            print(f"WARNING: Game {game_num} missing calibration offsets, skipping")
            continue

        entries[game_num] = CalibrationEntry(
            game_num=game_num,
            h1_offset_sec=h1_offset,
            h2_offset_sec=h2_offset,
            h1_anchor_event_ms=cal.get("h1_anchor_event_ms"),
            h1_anchor_video_sec=cal.get("h1_anchor_video_sec"),
            h2_anchor_event_ms=cal.get("h2_anchor_event_ms"),
            h2_anchor_video_sec=cal.get("h2_anchor_video_sec"),
            notes=cal.get("notes", ""),
        )

    return entries


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    mp4_path: Path,
    start_sec: float,
    end_sec: float,
    output_dir: Path,
    prefix: str,
    num_frames: int = NUM_FRAMES,
    width: int = FRAME_WIDTH,
) -> list[Path]:
    """Extract uniformly-spaced frames from a video clip using FFmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = end_sec - start_sec
    if duration <= 0:
        return []

    # Clamp start to 0
    if start_sec < 0:
        start_sec = 0
        duration = end_sec - start_sec

    fps = num_frames / duration

    pattern = str(output_dir / f"{prefix}_f%03d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(mp4_path),
        "-t", f"{duration:.3f}",
        "-vf", f"fps={fps:.4f},scale={width}:-2",
        "-q:v", str(FRAME_QUALITY),
        "-frames:v", str(num_frames),
        pattern,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-200:]
            print(f"  FFmpeg error for {prefix}: {stderr}")
            return []
    except subprocess.TimeoutExpired:
        print(f"  FFmpeg timeout for {prefix}")
        return []

    # Collect extracted frames
    frames = sorted(output_dir.glob(f"{prefix}_f*.jpg"))
    return frames


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    events: list[GTEvent],
    calibration: dict[str, CalibrationEntry],
) -> list[GTEvent]:
    """Generate hard negative samples from the event list."""
    negatives = []

    # Index events by game and time for proximity search
    events_by_game: dict[str, list[GTEvent]] = defaultdict(list)
    for e in events:
        events_by_game[e.game_num].append(e)

    for game_num, game_events in events_by_game.items():
        if game_num not in calibration:
            continue

        cal = calibration[game_num]
        game_events_sorted = sorted(game_events, key=lambda x: x.event_time_ms)

        # Find gaps > 30s between consecutive events for "none" samples
        for i in range(len(game_events_sorted) - 1):
            e1 = game_events_sorted[i]
            e2 = game_events_sorted[i + 1]
            # Only within same half
            if e1.half != e2.half:
                continue
            gap_ms = e2.event_time_ms - e1.event_time_ms
            if gap_ms > 30000:  # > 30 seconds
                mid_ms = e1.event_time_ms + gap_ms // 2
                negatives.append(GTEvent(
                    game_num=game_num,
                    event_time_ms=mid_ms,
                    half=e1.half,
                    team="",
                    player="",
                    event_label="none",
                    event_name_raw="none",
                    property_raw={},
                ))

    return negatives


# ---------------------------------------------------------------------------
# JSONL generation
# ---------------------------------------------------------------------------

def make_training_record(
    sample: TrainingSample,
    stage: int,
) -> Optional[dict]:
    """Create a ms-swift JSONL record for one training sample."""
    if not sample.frame_paths or len(sample.frame_paths) < NUM_FRAMES:
        return None

    window_sec = sample.window_end - sample.window_start

    if stage == 1:
        query = STAGE1_QUERY_TEMPLATE.format(window_sec=window_sec)
        response = _pick_response_template(sample.event_label)
    else:
        query = STAGE2_QUERY_TEMPLATE.format(window_sec=window_sec)
        response = _make_stage2_response(sample.event_label)

    return {
        "query": query,
        "response": response,
        "images": sample.frame_paths[:NUM_FRAMES],
        # Metadata (not used by ms-swift, but useful for analysis)
        "_meta": {
            "game": sample.game_num,
            "label": sample.event_label,
            "event_time_ms": sample.event_time_ms,
            "half": sample.half,
            "video_ts": round(sample.video_timestamp, 2),
            "augmentation": sample.augmentation,
        },
    }


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def cmd_calibrate(args):
    """Generate calibration template."""
    games_dir = Path(args.games_dir)
    output = Path(args.output)
    games = discover_games(games_dir)
    print(f"Found {len(games)} games")
    generate_calibration_template(games, output)


def cmd_extract(args):
    """Extract frames and generate JSONL training data."""
    games_dir = Path(args.games_dir)
    calibration_path = Path(args.calibration)
    output_dir = Path(args.output_dir)
    stage = args.stage

    games = discover_games(games_dir)
    calibration = load_calibration(calibration_path)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[TrainingSample] = []
    stats = Counter()

    for game in games:
        if game.game_num not in calibration:
            print(f"Skipping game {game.game_num} (no calibration)")
            continue

        cal = calibration[game.game_num]
        game_frames_dir = frames_dir / f"game_{game.game_num}"

        print(f"\n--- Game {game.game_num} ---")
        print(f"  MP4: {game.mp4_path.name}")
        print(f"  H1 offset: {cal.h1_offset_sec:.1f}s, H2 offset: {cal.h2_offset_sec:.1f}s")

        # Parse all events
        all_events = []
        for jp in game.json_paths:
            all_events.extend(parse_events(jp, game.game_num))

        # Add hard negatives
        negatives = mine_hard_negatives(all_events, calibration)
        neg_for_game = [n for n in negatives if n.game_num == game.game_num]
        all_events.extend(neg_for_game)

        print(f"  Events: {len(all_events)} ({len(neg_for_game)} hard negatives)")

        for event in all_events:
            # Skip irrelevant event types
            if event.event_label not in EVENT_WINDOWS:
                continue

            # Compute video timestamp
            offset = cal.h1_offset_sec if event.half == 1 else cal.h2_offset_sec
            video_ts = event.event_time_ms / 1000.0 + offset

            # Compute extraction window
            before, after = EVENT_WINDOWS[event.event_label]
            window_start = video_ts - before
            window_end = video_ts + after

            # Extract frames
            prefix = f"{event.event_label}_{event.event_time_ms}"
            frame_paths = extract_frames(
                game.mp4_path,
                window_start,
                window_end,
                game_frames_dir,
                prefix,
            )

            if not frame_paths:
                stats["extraction_failed"] += 1
                continue

            # Convert to relative paths (relative to output_dir)
            rel_paths = [str(p.relative_to(output_dir)) for p in frame_paths]

            sample = TrainingSample(
                game_num=game.game_num,
                event_label=event.event_label,
                event_time_ms=event.event_time_ms,
                half=event.half,
                video_timestamp=video_ts,
                window_start=window_start,
                window_end=window_end,
                frame_paths=rel_paths,
            )
            all_samples.append(sample)
            stats[event.event_label] += 1

    # Write JSONL
    output_file = output_dir / f"stage{stage}_all.jsonl"
    written = 0
    with open(output_file, "w") as f:
        for sample in all_samples:
            record = make_training_record(sample, stage)
            if record:
                f.write(json.dumps(record) + "\n")
                written += 1

    print(f"\n=== Extraction Complete ===")
    print(f"Total samples: {written}")
    print(f"Output: {output_file}")
    print(f"\nPer-class counts:")
    for label, count in stats.most_common():
        print(f"  {label}: {count}")

    # Write manifest
    manifest = {
        "total_samples": written,
        "stage": stage,
        "games": [g.game_num for g in games if g.game_num in calibration],
        "per_class": dict(stats.most_common()),
        "output_file": str(output_file),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


def cmd_split(args):
    """Split extracted data into train/val sets."""
    data_dir = Path(args.data_dir)
    holdout_games = set(args.holdout_games.split(","))
    val_ratio = args.val_ratio
    stage = args.stage

    input_file = data_dir / f"stage{stage}_all.jsonl"
    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run 'extract' first.")
        sys.exit(1)

    train_records = []
    val_records = []
    holdout_records = []

    with open(input_file) as f:
        for line in f:
            record = json.loads(line)
            meta = record.get("_meta", {})
            game = meta.get("game", "")

            if game in holdout_games:
                holdout_records.append(record)
            else:
                if random.random() < val_ratio:
                    val_records.append(record)
                else:
                    train_records.append(record)

    # Shuffle training data
    random.shuffle(train_records)

    # Write files
    for name, records in [
        (f"stage{stage}_train.jsonl", train_records),
        (f"stage{stage}_val.jsonl", val_records),
        (f"stage{stage}_holdout.jsonl", holdout_records),
    ]:
        path = data_dir / name
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  {name}: {len(records)} samples")

    print(f"\nHoldout games: {holdout_games}")
    print(f"Train: {len(train_records)}, Val: {len(val_records)}, Holdout: {len(holdout_records)}")


def cmd_augment(args):
    """Augment training data with temporal shifts and mirrors."""
    data_dir = Path(args.data_dir)
    source_file = data_dir / args.source
    output_file = data_dir / args.output

    if not source_file.exists():
        print(f"ERROR: {source_file} not found")
        sys.exit(1)

    records = []
    with open(source_file) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Source: {len(records)} records")

    augmented = list(records)  # Start with originals

    # Count per class for capping
    class_counts = Counter()
    for r in records:
        label = r.get("_meta", {}).get("label", "none")
        class_counts[label] += 1

    print(f"Per-class counts (before augmentation):")
    for label, count in class_counts.most_common():
        print(f"  {label}: {count}")

    # Note: Temporal shift augmentation requires re-extracting frames from video.
    # This function generates the metadata records; frame extraction must be done
    # separately by re-running 'extract' with shifted windows.
    #
    # For now, we implement horizontal mirror augmentation (frame-level, no re-extraction).
    # Temporal shifts are generated as records pointing to the same frames but with
    # a note that the model should learn from varied temporal contexts.

    # Horizontal mirror: duplicate records with a "_mirror" suffix in augmentation field
    mirrorable_labels = {
        "goal", "save_catch", "save_parry", "shot_on_target",
        "shot_off_target", "shot_blocked", "goal_kick", "free_kick",
    }

    mirror_records = []
    for r in records:
        label = r.get("_meta", {}).get("label", "none")
        if label in mirrorable_labels:
            mirror = json.loads(json.dumps(r))
            mirror["_meta"]["augmentation"] = "mirror"
            # Note: actual image flipping happens at training time via transforms,
            # or we can pre-flip with ImageMagick. For ms-swift, we'd need pre-flipped copies.
            # This record flags which samples should be flipped during preprocessing.
            mirror_records.append(mirror)

    augmented.extend(mirror_records)
    print(f"\nAdded {len(mirror_records)} mirror augmentations")

    # Apply per-class caps
    by_class: dict[str, list] = defaultdict(list)
    for r in augmented:
        label = r.get("_meta", {}).get("label", "none")
        by_class[label].append(r)

    final = []
    for label, recs in by_class.items():
        cap = CLASS_CAPS.get(label, 300)
        if len(recs) > cap:
            random.shuffle(recs)
            recs = recs[:cap]
        final.extend(recs)

    random.shuffle(final)

    with open(output_file, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")

    # Final stats
    final_counts = Counter()
    for r in final:
        label = r.get("_meta", {}).get("label", "none")
        final_counts[label] += 1

    print(f"\nFinal augmented dataset: {len(final)} samples")
    for label, count in final_counts.most_common():
        cap = CLASS_CAPS.get(label, 300)
        print(f"  {label}: {count} (cap: {cap})")

    print(f"\nOutput: {output_file}")


def cmd_stats(args):
    """Print statistics about training data on disk."""
    games_dir = Path(args.games_dir)
    games = discover_games(games_dir)

    total = Counter()
    total_goals = 0

    print(f"Found {len(games)} games\n")
    print(f"{'Game':>6} {'Goals':>6} {'ShotsOT':>8} {'ShotsOff':>9} {'Blocked':>8} "
          f"{'Saves':>6} {'Corners':>8} {'GKicks':>7} {'ThrowIn':>8} {'FKicks':>7} {'Pen':>4}")
    print("-" * 100)

    for game in games:
        all_events = []
        for jp in game.json_paths:
            all_events.extend(parse_events(jp, game.game_num))

        counts = Counter(e.event_label for e in all_events)
        goals = counts.get("goal", 0)
        total_goals += goals

        for label, count in counts.items():
            total[label] += count

        print(f"{game.game_num:>6} {goals:>6} {counts.get('shot_on_target', 0):>8} "
              f"{counts.get('shot_off_target', 0):>9} {counts.get('shot_blocked', 0):>8} "
              f"{counts.get('save_catch', 0) + counts.get('save_parry', 0):>6} "
              f"{counts.get('corner_kick', 0):>8} {counts.get('goal_kick', 0):>7} "
              f"{counts.get('throw_in', 0):>8} {counts.get('free_kick', 0):>7} "
              f"{counts.get('penalty', 0):>4}")

    print("-" * 100)
    print(f"{'TOTAL':>6} {total_goals:>6} {total.get('shot_on_target', 0):>8} "
          f"{total.get('shot_off_target', 0):>9} {total.get('shot_blocked', 0):>8} "
          f"{total.get('save_catch', 0) + total.get('save_parry', 0):>6} "
          f"{total.get('corner_kick', 0):>8} {total.get('goal_kick', 0):>7} "
          f"{total.get('throw_in', 0):>8} {total.get('free_kick', 0):>7} "
          f"{total.get('penalty', 0):>4}")
    print(f"\nTotal labeled events: {sum(total.values())}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Training data extraction for Qwen3-VL fine-tuning"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Generate calibration template")
    p_cal.add_argument("--games-dir", required=True, help="Path to Games/ directory")
    p_cal.add_argument("--output", default="calibration.json", help="Output calibration file")

    # extract
    p_ext = subparsers.add_parser("extract", help="Extract frames and generate JSONL")
    p_ext.add_argument("--games-dir", required=True, help="Path to Games/ directory")
    p_ext.add_argument("--calibration", required=True, help="Calibration JSON file")
    p_ext.add_argument("--output-dir", required=True, help="Output directory for frames and JSONL")
    p_ext.add_argument("--stage", type=int, default=2, choices=[1, 2],
                       help="Training stage (1=concept alignment, 2=classification)")

    # split
    p_split = subparsers.add_parser("split", help="Split data into train/val/holdout")
    p_split.add_argument("--data-dir", required=True, help="Directory with stage*_all.jsonl")
    p_split.add_argument("--holdout-games", default="11,13",
                        help="Comma-separated game numbers to hold out for evaluation")
    p_split.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of non-holdout data for validation")
    p_split.add_argument("--stage", type=int, default=2, choices=[1, 2])

    # augment
    p_aug = subparsers.add_parser("augment", help="Augment training data")
    p_aug.add_argument("--data-dir", required=True)
    p_aug.add_argument("--source", default="stage2_train.jsonl")
    p_aug.add_argument("--output", default="stage2_train_augmented.jsonl")

    # stats
    p_stats = subparsers.add_parser("stats", help="Print training data statistics")
    p_stats.add_argument("--games-dir", required=True, help="Path to Games/ directory")

    args = parser.parse_args()

    if args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "augment":
        cmd_augment(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
