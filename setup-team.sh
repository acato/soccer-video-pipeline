#!/usr/bin/env bash
# setup-team.sh — Save your team name and jersey colors.
# Config is stored at ~/.soccer-pipeline-team.json so it survives project re-downloads.
#
# Usage:
#   ./setup-team.sh "Team Name" --kit Home blue teal --kit Away white neon_yellow
#   ./setup-team.sh --add-kit Third dark_blue neon_green
#   ./setup-team.sh --colors
#   ./setup-team.sh              (shows current config)
set -euo pipefail

CONFIG_FILE="${TEAM_CONFIG_PATH:-$HOME/.soccer-pipeline-team.json}"

# ── Available colors ────────────────────────────────────────────────────
COLORS=(
  "white"        "⬜"
  "silver"       "◻️"
  "gray"         "⬛"
  "black"        "◼️"
  "red"          "🟥"
  "dark_red"     "🟥"
  "maroon"       "🟫"
  "burgundy"     "🟫"
  "orange"       "🟧"
  "neon_orange"  "🟧"
  "yellow"       "🟨"
  "neon_yellow"  "🟨"
  "green"        "🟩"
  "dark_green"   "🟩"
  "neon_green"   "🟩"
  "teal"         "🟦"
  "sky_blue"     "🟦"
  "light_blue"   "🟦"
  "blue"         "🟦"
  "dark_blue"    "🟦"
  "navy"         "🟦"
  "purple"       "🟪"
  "pink"         "🟪"
  "hot_pink"     "🟪"
  "neon_pink"    "🟪"
)

COLOR_NAMES=()
for ((i = 0; i < ${#COLORS[@]}; i += 2)); do
  COLOR_NAMES+=("${COLORS[$i]}")
done

valid_color() {
  local c="$1"
  for name in "${COLOR_NAMES[@]}"; do
    [[ "$name" == "$c" ]] && return 0
  done
  return 1
}

show_colors() {
  echo ""
  echo "Available jersey colors:"
  echo "────────────────────────"
  for ((i = 0; i < ${#COLORS[@]}; i += 2)); do
    printf "  %s  %-14s\n" "${COLORS[$((i + 1))]}" "${COLORS[$i]}"
  done
  echo ""
}

show_config() {
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "No team set up yet."
    echo "Run: ./setup-team.sh \"Your Team Name\" --kit Home blue teal"
    return
  fi
  echo ""
  echo "Current team config ($CONFIG_FILE):"
  echo "────────────────────────────────────"
  # Use python for pretty-printing since it's always available
  python3 -c "import json, sys; data=json.load(open(sys.argv[1])); print(json.dumps(data, indent=2))" "$CONFIG_FILE"
  echo ""
}

# ── No arguments: show current config ──────────────────────────────────
if [[ $# -eq 0 ]]; then
  show_config
  exit 0
fi

# ── --colors: show palette ─────────────────────────────────────────────
if [[ "$1" == "--colors" ]]; then
  show_colors
  exit 0
fi

# ── --add-kit: add a kit to existing config ────────────────────────────
if [[ "$1" == "--add-kit" ]]; then
  if [[ $# -ne 4 ]]; then
    echo "Usage: ./setup-team.sh --add-kit KitName outfield_color gk_color"
    echo "Example: ./setup-team.sh --add-kit Third dark_blue neon_green"
    exit 1
  fi
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: No team config found. Set up your team first:"
    echo "  ./setup-team.sh \"Team Name\" --kit Home blue teal"
    exit 1
  fi
  kit_name="$2"
  outfield="$3"
  gk="$4"
  if ! valid_color "$outfield"; then
    echo "Error: Unknown color '$outfield'. Run ./setup-team.sh --colors to see options."
    exit 1
  fi
  if ! valid_color "$gk"; then
    echo "Error: Unknown color '$gk'. Run ./setup-team.sh --colors to see options."
    exit 1
  fi
  python3 -c "
import json, sys
path, kit, out, gk = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
data = json.load(open(path))
data['kits'][kit] = {'outfield_color': out, 'gk_color': gk}
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
print(f'Added kit \"{kit}\" to {data[\"team_name\"]}.')
" "$CONFIG_FILE" "$kit_name" "$outfield" "$gk"
  show_config
  exit 0
fi

# ── Full setup: "TeamName" --kit ... --kit ... ─────────────────────────
team_name="$1"
shift

if [[ $# -eq 0 ]]; then
  echo "Error: You need at least one --kit."
  echo "Usage: ./setup-team.sh \"Team Name\" --kit Home blue teal"
  exit 1
fi

# Parse --kit arguments
declare -A kits_outfield
declare -A kits_gk
kit_order=()

while [[ $# -gt 0 ]]; do
  if [[ "$1" != "--kit" ]]; then
    echo "Error: Expected --kit, got '$1'"
    echo "Usage: ./setup-team.sh \"Team Name\" --kit Home blue teal --kit Away white neon_yellow"
    exit 1
  fi
  shift
  if [[ $# -lt 3 ]]; then
    echo "Error: --kit needs three values: KitName outfield_color gk_color"
    echo "Example: --kit Home blue teal"
    exit 1
  fi
  kit_name="$1"
  outfield="$2"
  gk="$3"
  shift 3

  if ! valid_color "$outfield"; then
    echo "Error: Unknown color '$outfield'. Run ./setup-team.sh --colors to see options."
    exit 1
  fi
  if ! valid_color "$gk"; then
    echo "Error: Unknown color '$gk'. Run ./setup-team.sh --colors to see options."
    exit 1
  fi

  kits_outfield["$kit_name"]="$outfield"
  kits_gk["$kit_name"]="$gk"
  kit_order+=("$kit_name")
done

if [[ ${#kit_order[@]} -eq 0 ]]; then
  echo "Error: No kits provided. Use --kit Home blue teal"
  exit 1
fi

# Build JSON via python (portable, no jq dependency)
kits_json="{"
first=true
for kit_name in "${kit_order[@]}"; do
  if $first; then first=false; else kits_json+=","; fi
  kits_json+="\"$kit_name\":{\"outfield_color\":\"${kits_outfield[$kit_name]}\",\"gk_color\":\"${kits_gk[$kit_name]}\"}"
done
kits_json+="}"

python3 -c "
import json, sys
team_name = sys.argv[1]
kits = json.loads(sys.argv[2])
data = {'team_name': team_name, 'kits': kits}
with open(sys.argv[3], 'w') as f:
    json.dump(data, f, indent=2)
print(f'Saved team config for \"{team_name}\" to {sys.argv[3]}')
" "$team_name" "$kits_json" "$CONFIG_FILE"

show_config
echo "You're all set! Open http://localhost:8080/ui to process a game."
