#!/usr/bin/env bash
# Run a multi-game eval against the currently-deployed model.
#
# Usage:
#   bash multi_game_eval.sh RUN_NUMBER COMMIT_SHA "v7 baseline notes"
#
# Submits jobs for: Rush (game_11), Game 20, Game 21, Game 22 — sequentially.
# Each job waits for the previous to complete (vLLM is single-endpoint).
# After each, runs evaluate_detection.py with the right offsets.
# Builds an aggregated table at the end.
#
# Run on Mac:
#   bash scripts/multi_game_eval.sh 67 a10ab00 "v7 c250 first eval"
set -euo pipefail

RUN_BASE="${1:?run number required, e.g. 67}"
COMMIT="${2:?commit sha required}"
NOTES="${3:-multi-game eval}"

API="http://127.0.0.1:8088"
GT_ROOT="$HOME/soccer-runs/gt"
STATE_DIR="$HOME/soccer-runs/state"
SUMMARY_FILE="$STATE_DIR/multi_game_run_${RUN_BASE}_summary.json"
mkdir -p "$STATE_DIR"

# Game configs: game_id, video_filename, gt_subdir, video-offset, half2-start, half2-game-offset, h1_json, h2_json
declare -a GAMES=(
  "rush|2026-02-07 - Rush - GA2008.mp4||418.0|3916.0|2700|/Users/aless/soccer-runs/gt/08 GA (U19) vs Washington Rush U19 (W)_1st Half.json|/Users/aless/soccer-runs/gt/08 GA (U19) vs Washington Rush U19 (W)_2nd Half.json"
  "game_20|2026-04-18 Celtic - Reign GA 11.mp4|game20|124.0|3554.0|2400|2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_1st Half.json|2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_2nd Half.json"
  "game_21|2026-04-25 Eastern WA Surf - Reign GA11.mp4|game21|250.0|4130.0|2400|2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_1st Half.json|2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_2nd Half.json"
  "game_22|2026-04-26 Spokane Shadow - Reign GA11.mp4|game22|90.0|2900.0|2700|2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_1st Half.json|2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_2nd Half.json"
)

# Known team configs (best-guess colors; doesn't affect eval F1)
declare -A MATCH_CONFIGS=(
  [rush]='{"team":{"team_name":"Rush","outfield_color":"white","gk_color":"neon_yellow"},"opponent":{"team_name":"GA 2008","outfield_color":"blue","gk_color":"neon_green"}}'
  [game_20]='{"team":{"team_name":"Reign 2011 GA","outfield_color":"red","gk_color":"neon_yellow"},"opponent":{"team_name":"Seattle Celtic U15","outfield_color":"green","gk_color":"neon_green"}}'
  [game_21]='{"team":{"team_name":"Reign 2011 GA","outfield_color":"red","gk_color":"neon_yellow"},"opponent":{"team_name":"East WA Surf U15","outfield_color":"blue","gk_color":"neon_green"}}'
  [game_22]='{"team":{"team_name":"Reign 2011 GA","outfield_color":"red","gk_color":"neon_yellow"},"opponent":{"team_name":"Spokane Shadow U15","outfield_color":"green","gk_color":"neon_green"}}'
)

EVALS=()

idx=0
for entry in "${GAMES[@]}"; do
  IFS='|' read -r game_id video gt_subdir voff h2start h2gameoff h1 h2 <<< "$entry"
  run_n=$((RUN_BASE + idx))
  echo
  echo "============================================================"
  echo "Run $run_n: $game_id ($video)"
  echo "============================================================"

  # Resolve full GT paths
  if [ -n "$gt_subdir" ]; then
    h1="$GT_ROOT/$gt_subdir/$h1"
    h2="$GT_ROOT/$gt_subdir/$h2"
  fi
  for f in "$h1" "$h2"; do
    if [ ! -f "$f" ]; then
      echo "ERROR: GT file not found: $f" >&2
      exit 1
    fi
  done

  # Build job_payload
  mc="${MATCH_CONFIGS[$game_id]}"
  payload=$(python3 -c "
import json
print(json.dumps({
  'nas_path': '''$video''',
  'reel_types': ['keeper', 'highlights'],
  'audio_fusion_enabled': False,
  'temporal_fusion_enabled': False,
  'kickoff_verifier_enabled': True,
  'match_config': $mc,
}))
")

  # Submit job
  resp=$(curl -s -X POST "$API/jobs" -H "Content-Type: application/json" -d "$payload")
  job_id=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
  status=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  echo "  job_id=$job_id status=$status"

  # If it returned an existing complete job (dedupe), delete + resubmit
  if [ "$status" = "complete" ]; then
    echo "  dedupe hit on existing job; deleting + resubmitting..."
    curl -s -X DELETE "$API/jobs/$job_id" >/dev/null
    resp=$(curl -s -X POST "$API/jobs" -H "Content-Type: application/json" -d "$payload")
    job_id=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
    echo "  resubmitted job_id=$job_id"
  fi

  # Poll until complete
  echo "  waiting for completion..."
  while :; do
    s=$(curl -s "$API/jobs/$job_id" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status','?'),round(d.get('progress_pct',0),1))" 2>/dev/null || echo "? 0")
    state=$(echo "$s" | awk '{print $1}')
    pct=$(echo "$s" | awk '{print $2}')
    echo "    [$(date +%H:%M:%S)] $game_id status=$state progress=${pct}%"
    if [ "$state" = "complete" ] || [ "$state" = "failed" ]; then
      break
    fi
    sleep 60
  done

  # Run eval
  events_file="/tmp/soccer-pipeline/$job_id/diagnostics/dual_pass_events.jsonl"
  eval_out="$STATE_DIR/run_${run_n}_eval.json"
  echo "  evaluating..."
  $HOME/Downloads/soccer-video-pipeline/.venv/bin/python \
    $HOME/Downloads/soccer-video-pipeline/scripts/evaluate_detection.py \
    "$events_file" \
    --tolerance 45 \
    --video-offset "$voff" \
    --half2-start "$h2start" \
    --half2-game-offset "$h2gameoff" \
    --gt-file "$h1" --gt-file "$h2" \
    --json-out "$eval_out"

  # Capture eval result
  f1=$(python3 -c "import json; print(round(json.load(open('$eval_out'))['overall']['f1'],3))")
  echo "  $game_id F1=$f1  (eval: $eval_out)"
  EVALS+=("$game_id|$run_n|$job_id|$f1|$eval_out")
  idx=$((idx + 1))
done

# Aggregated summary
echo
echo "============================================================"
echo "Multi-game summary (run base $RUN_BASE, commit $COMMIT)"
echo "============================================================"
python3 <<PYEOF > "$SUMMARY_FILE"
import json
evals = "$(printf '%s,' "${EVALS[@]}")".strip(",").split(",")
out = {"run_base": $RUN_BASE, "commit_sha": "$COMMIT", "notes": "$NOTES", "games": []}
totals = {"tp":0, "fn":0, "fp":0}
for e in evals:
    if not e: continue
    game_id, run_n, job_id, f1, eval_out = e.split("|")
    d = json.load(open(eval_out))
    o = d["overall"]
    out["games"].append({
        "game_id": game_id, "run_number": int(run_n), "job_id": job_id,
        "f1": float(f1), "tp": o["tp"], "fn": o["fn"], "fp": o["fp"],
        "precision": o["precision"], "recall": o["recall"],
        "goal_recall": d["per_type"].get("goal",{}).get("recall", 0.0),
        "goal_precision": d["per_type"].get("goal",{}).get("precision", 0.0),
    })
    totals["tp"] += o["tp"]; totals["fn"] += o["fn"]; totals["fp"] += o["fp"]
p = totals["tp"]/max(1,totals["tp"]+totals["fp"])
r = totals["tp"]/max(1,totals["tp"]+totals["fn"])
out["aggregated"] = {**totals, "precision": p, "recall": r,
                     "f1": (2*p*r)/max(1e-9,p+r)}
print(json.dumps(out, indent=2))
PYEOF
cat "$SUMMARY_FILE"
echo
echo "Wrote summary to $SUMMARY_FILE"
