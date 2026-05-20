#!/usr/bin/env bash
# Produce per-game tiered events:
#   GOAL tiers:
#     confirmed = dual_pass detector goals (precision ~0.88)
#     candidate = ensemble GOALs not within 60s of a confirmed (recall booster)
#   SAVE tiers (added 2026-05-20):
#     confirmed = catch + shot_stop_* events (precision ~0.43)
#     candidate = shot_on_target + free_kick_shot not within 60s of confirmed save
#                 or any goal (recall booster: 0.40 -> ~0.79)
#
# Reads:
#   - /tmp/soccer-pipeline/<job_id>/events.jsonl (dual_pass detector output)
#   - /tmp/kickoff_<game>_formation_v2_base.jsonl (ensemble formation pass)
#   - /tmp/kickoff_<game>_pattern_v11_0191.jsonl (ensemble pattern pass — rush only)
#
# Writes /tmp/kickoff_<game>_tiered_events.jsonl per game.
#
# Uses case statements rather than `declare -A` so it runs under bash 3.2 (macOS).

set -euo pipefail
PY=~/Downloads/soccer-video-pipeline/.venv/bin/python
SCRIPTS=~/Downloads/soccer-video-pipeline/scripts

if [[ $# -gt 0 ]]; then
  GAMES=("$@")
else
  GAMES=(game_20 game_22 game_21 rush)
fi

for game in "${GAMES[@]}"; do
  echo "=== $game ==="
  case $game in
    game_20)
      dual_pass=/tmp/soccer-pipeline/a0f8f93c-8611-466b-934c-8cd48a2aee00/events.jsonl
      ensembles=(--ensemble /tmp/kickoff_game20_1H_formation_base.jsonl
                 --ensemble /tmp/kickoff_game20_2H_formation_base.jsonl)
      ;;
    game_22)
      dual_pass=/tmp/soccer-pipeline/031ee71a-7a4f-4605-a642-9bff003e4804/events.jsonl
      ensembles=(--ensemble /tmp/kickoff_game_22_formation_v2_base.jsonl)
      ;;
    game_21)
      dual_pass=/tmp/soccer-pipeline/60ed91b3-ebcb-4ddb-ac18-768923a17419/events.jsonl
      ensembles=(--ensemble /tmp/kickoff_game_21_formation_v2_base.jsonl)
      ;;
    rush)
      dual_pass=/tmp/soccer-pipeline/47d1358c-268c-4577-b4e1-f9943f51be6a/events.jsonl
      ensembles=(--ensemble /tmp/kickoff_rush_formation_v2_base.jsonl
                 --ensemble /tmp/kickoff_rush_pattern_v11_0191.jsonl)
      ;;
    *)
      echo "  unknown game: $game" >&2; exit 1 ;;
  esac
  out=/tmp/kickoff_${game}_tiered_events.jsonl
  $PY $SCRIPTS/merge_ensemble_into_events.py \
    --dual-pass "$dual_pass" \
    "${ensembles[@]}" \
    --relaxed-aggregation \
    --negative-evidence \
    --save-tiers \
    --out "$out"
done
