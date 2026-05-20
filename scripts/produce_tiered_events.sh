#!/usr/bin/env bash
# Produce per-game tiered goal events:
#   confirmed tier = dual_pass detector goals (precision ~0.88)
#   candidate tier = ensemble GOALs not within 60s of a confirmed (recall booster)
#
# Reads from existing artifacts:
#   - /tmp/soccer-pipeline/<job_id>/events.jsonl (dual_pass detector output)
#   - /tmp/kickoff_<game>_formation_v2_base.jsonl (ensemble formation pass)
#   - /tmp/kickoff_<game>_pattern_v11_0191.jsonl (ensemble pattern pass — rush only)
#
# Writes /tmp/kickoff_<game>_tiered_events.jsonl per game.

set -euo pipefail
PY=~/Downloads/soccer-video-pipeline/.venv/bin/python
SCRIPTS=~/Downloads/soccer-video-pipeline/scripts

declare -A DUAL_PASS_EVENTS=(
  [game_20]="/tmp/soccer-pipeline/a0f8f93c-8611-466b-934c-8cd48a2aee00/events.jsonl"
  [game_22]="/tmp/soccer-pipeline/031ee71a-7a4f-4605-a642-9bff003e4804/events.jsonl"
  [game_21]="/tmp/soccer-pipeline/60ed91b3-ebcb-4ddb-ac18-768923a17419/events.jsonl"
  [rush]="/tmp/soccer-pipeline/47d1358c-268c-4577-b4e1-f9943f51be6a/events.jsonl"
)

declare -A FORMATION_FILES=(
  [game_20]="/tmp/kickoff_game20_1H_formation_base.jsonl /tmp/kickoff_game20_2H_formation_base.jsonl"
  [game_22]="/tmp/kickoff_game_22_formation_v2_base.jsonl"
  [game_21]="/tmp/kickoff_game_21_formation_v2_base.jsonl"
  [rush]="/tmp/kickoff_rush_formation_v2_base.jsonl"
)

declare -A PATTERN_FILES=(
  [rush]="/tmp/kickoff_rush_pattern_v11_0191.jsonl"
)

GAMES=("$@")
[[ ${#GAMES[@]} -eq 0 ]] && GAMES=(game_20 game_22 game_21 rush)

for game in "${GAMES[@]}"; do
  echo "=== $game ==="
  dual_pass=${DUAL_PASS_EVENTS[$game]}
  out=/tmp/kickoff_${game}_tiered_events.jsonl
  ensemble_args=""
  for f in ${FORMATION_FILES[$game]}; do
    ensemble_args="$ensemble_args --ensemble $f"
  done
  if [[ -n ${PATTERN_FILES[$game]:-} ]]; then
    ensemble_args="$ensemble_args --ensemble ${PATTERN_FILES[$game]}"
  fi
  $PY $SCRIPTS/merge_ensemble_into_events.py \
    --dual-pass "$dual_pass" \
    $ensemble_args \
    --out "$out"
done
