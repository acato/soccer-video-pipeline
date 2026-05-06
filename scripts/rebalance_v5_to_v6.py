#!/usr/bin/env python3
"""Rebalance v5 LoRA dataset → v6 to fix the catastrophic forgetting that
Run 60/61 exposed (catch / goal / shot_stop_diving / penalty all collapsed
to F1=0 because v5 had <1% of windows for each).

What this does:
  1. Reads v5 per-game label JSONLs from --v5-dir/labels/.
  2. For TRAIN windows:
     a. Drops penalty entirely (5 examples, hopeless to learn).
     b. Carves out 'hard-negative' nones (within ±90s of any shot/goal in
        the same game) and keeps ALL of them.
     c. Subsamples the remaining 'none' windows down to --none-cap.
     d. Oversamples rare-class windows by duplicating each row N times
        (simple, no new frame extraction). N is per-class via --oversample.
     e. Keeps frequent-class windows 1×.
  3. For EVAL windows: keep as-is (no rebalancing — it would hide bias).
  4. Writes v6 per-game label JSONLs to --v6-dir/labels/. Frame paths are
     kept relative (matching v5 format); the downstream
     convert_v5_to_swift.py call must use
     --frames-prefix /mnt/transit/soccer-finetune/lora_dataset_v5/
     so the relative paths resolve to the v5 frame files.
  5. Writes manifest with the before/after class distribution.

Usage:
  python scripts/rebalance_v5_to_v6.py \\
      --v5-dir /mnt/transit/soccer-finetune/lora_dataset_v5 \\
      --v6-dir /mnt/transit/soccer-finetune/lora_dataset_v6
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Defaults — based on v5 analysis (see ql3_training_run.md / Run 60+61):
#   v5 had 8995 'none' (73.6%) vs 5 'penalty' (0.04%). LoRA collapsed all
#   classes <1% prevalence to zero detections.
# ─────────────────────────────────────────────────────────────────────────
DEFAULT_NONE_CAP = 3000          # ~3:1 none-to-event balance instead of 2.8:1
DEFAULT_HARD_NEG_WINDOW = 90.0   # nones within ±90s of a shot/goal stay
DEFAULT_OVERSAMPLE = {
    # class : duplication factor (1 = no oversample)
    "catch": 8,             # 199 → ~1600
    "goal": 18,             # 85  → ~1500
    "shot_stop_diving": 21, # 70  → ~1500
    "shot_on_target": 2,    # 752 → ~1500 (was collapsing despite plenty of data)
    "free_kick_shot": 2,    # 376 → ~750
    "corner_kick": 2,       # 201 → ~400
    # NOT oversampled: throw_in (1571 plenty), goal_kick (479 ok)
    # DROPPED: penalty (5 examples — keep frozen-base behavior)
}
DROPPED_CLASSES = {"penalty"}
SHOT_LIKE_FOR_HARD_NEG = {"shot_on_target", "goal", "free_kick_shot"}


def load_v5_records(v5_labels_dir: Path) -> dict[str, list[dict]]:
    """Per-game list of records, in original window order."""
    by_game: dict[str, list[dict]] = {}
    for jp in sorted(v5_labels_dir.glob("*.jsonl")):
        with open(jp) as f:
            by_game[jp.stem] = [json.loads(line) for line in f if line.strip()]
    return by_game


def categorize_window(rec: dict) -> str:
    """Return the dominant event_type in the window, or 'none'."""
    types = {ev["event_type"] for ev in rec["target"]}
    types.discard("none")
    if not types:
        return "none"
    # If multiple events in one window, pick the rarest (most informative).
    rarity_order = ["penalty", "shot_stop_diving", "goal", "catch",
                    "corner_kick", "free_kick_shot", "shot_on_target",
                    "goal_kick", "throw_in"]
    for cls in rarity_order:
        if cls in types:
            return cls
    return next(iter(types))


def find_hard_negative_nones(records: list[dict], window_sec: float) -> set[int]:
    """Return v5 window indices for 'none' windows that fall within
    ±window_sec of any shot-like event in this game."""
    shot_times: list[float] = []
    for r in records:
        for ev in r["target"]:
            if ev["event_type"] in SHOT_LIKE_FOR_HARD_NEG:
                shot_times.append((r["window_start_sec"] + r["window_end_sec"]) / 2)
                break
    if not shot_times:
        return set()
    shot_times.sort()
    out = set()
    for r in records:
        if categorize_window(r) != "none":
            continue
        center = (r["window_start_sec"] + r["window_end_sec"]) / 2
        for st in shot_times:
            if abs(st - center) <= window_sec:
                out.add(r["window_idx"])
                break
    return out


def rebalance_train_split(records: list[dict], *,
                           none_cap: int, hard_neg_window: float,
                           oversample: dict[str, int],
                           rng: random.Random,
                           ) -> tuple[list[dict], dict]:
    """Apply the rebalancing rules to a single train game's records.
    Returns (output_records, stats_dict)."""
    hard_neg_idxs = find_hard_negative_nones(records, hard_neg_window)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        cls = categorize_window(r)
        if cls in DROPPED_CLASSES:
            continue
        if cls == "none" and r["window_idx"] in hard_neg_idxs:
            by_class["none_hard_neg"].append(r)
        else:
            by_class[cls].append(r)

    out: list[dict] = []
    stats = {"in": Counter(), "out": Counter(), "hard_neg": len(hard_neg_idxs)}
    for cls, recs in by_class.items():
        stats["in"][cls] = len(recs)
        if cls == "none":
            # Subsample regular nones evenly across the game's timeline so
            # we don't lose entire halves.
            n = min(none_cap, len(recs))
            keep = rng.sample(recs, n) if n < len(recs) else list(recs)
            out.extend(keep)
            stats["out"][cls] = len(keep)
        elif cls == "none_hard_neg":
            # Keep all hard negatives.
            out.extend(recs)
            stats["out"][cls] = len(recs)
        else:
            mult = oversample.get(cls, 1)
            for _ in range(mult):
                out.extend(recs)
            stats["out"][cls] = len(recs) * mult
    rng.shuffle(out)
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v5-dir", required=True,
                    help="Path to existing lora_dataset_v5/ directory")
    ap.add_argument("--v6-dir", required=True,
                    help="Output directory for lora_dataset_v6/")
    ap.add_argument("--none-cap", type=int, default=DEFAULT_NONE_CAP,
                    help=f"Per-game cap on 'none' windows (default {DEFAULT_NONE_CAP} total)")
    ap.add_argument("--hard-neg-window", type=float, default=DEFAULT_HARD_NEG_WINDOW,
                    help=f"Seconds around shot-like events for hard negatives (default {DEFAULT_HARD_NEG_WINDOW})")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v5_dir = Path(args.v5_dir)
    v6_dir = Path(args.v6_dir)
    v5_labels = v5_dir / "labels"
    if not v5_labels.is_dir():
        raise SystemExit(f"missing {v5_labels}")
    v6_labels = v6_dir / "labels"
    v6_labels.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    by_game = load_v5_records(v5_labels)
    print(f"loaded {sum(len(v) for v in by_game.values())} v5 records "
          f"across {len(by_game)} games")

    # Distribute the global none-cap across train games proportional to game length.
    train_games = [g for g, recs in by_game.items() if recs and recs[0]["split"] == "train"]
    eval_games = [g for g, recs in by_game.items() if recs and recs[0]["split"] == "eval"]
    train_none_totals = {
        g: sum(1 for r in by_game[g] if categorize_window(r) == "none")
        for g in train_games
    }
    grand_total_none = sum(train_none_totals.values())
    per_game_cap = {
        g: max(50, int(args.none_cap * train_none_totals[g] / max(1, grand_total_none)))
        for g in train_games
    }

    overall_in: Counter = Counter()
    overall_out: Counter = Counter()
    per_game_stats = {}
    for game in sorted(by_game.keys()):
        recs = by_game[game]
        if not recs:
            continue
        if recs[0]["split"] == "eval":
            # Eval set: leave alone, just rewrite frame paths so v6 is self-contained.
            out_recs = list(recs)  # frame paths kept relative
            per_game_stats[game] = {
                "split": "eval", "in": len(recs), "out": len(out_recs),
            }
            in_dist = Counter(categorize_window(r) for r in recs)
            for c, n in in_dist.items():
                overall_in[c] += n
                overall_out[c] += n
        else:
            balanced, stats = rebalance_train_split(
                recs, none_cap=per_game_cap[game],
                hard_neg_window=args.hard_neg_window,
                oversample=DEFAULT_OVERSAMPLE, rng=rng,
            )
            out_recs = balanced  # frame paths kept relative
            per_game_stats[game] = {"split": "train", **stats,
                                     "out": dict(stats["out"]),
                                     "in": dict(stats["in"])}
            for c, n in stats["in"].items():
                overall_in[c] += n
            for c, n in stats["out"].items():
                overall_out[c] += n

        out_path = v6_labels / f"{game}.jsonl"
        with open(out_path, "w") as fh:
            for r in out_recs:
                fh.write(json.dumps(r) + "\n")
        print(f"  {game} ({recs[0]['split']:5s}): "
              f"{len(recs):5d} in → {len(out_recs):5d} out  ({out_path.name})")

    manifest = {
        "v5_dir": str(v5_dir),
        "v6_dir": str(v6_dir),
        "none_cap_total": args.none_cap,
        "hard_neg_window_sec": args.hard_neg_window,
        "oversample_factors": DEFAULT_OVERSAMPLE,
        "dropped_classes": sorted(DROPPED_CLASSES),
        "per_game_none_cap": per_game_cap,
        "per_game": per_game_stats,
        "overall_class_distribution_in": dict(overall_in),
        "overall_class_distribution_out": dict(overall_out),
    }
    (v6_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"{'class':<24}{'v5 in':>10}{'v6 out':>10}{'mult':>8}")
    print("-" * 52)
    for cls in sorted(set(overall_in) | set(overall_out)):
        i = overall_in.get(cls, 0)
        o = overall_out.get(cls, 0)
        mult = f"{o/i:.1f}x" if i else "-"
        print(f"{cls:<24}{i:>10d}{o:>10d}{mult:>8}")
    print()
    print(f"v6 manifest at {v6_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
