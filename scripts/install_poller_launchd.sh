#!/usr/bin/env bash
#
# install_poller_launchd.sh — install the run poller as a launchd user agent.
#
# Run this once on the Mac after cloning/updating the repo. It:
#   1. Creates ~/soccer-runs/{state,logs}
#   2. Renders infra/com.soccer.runpoller.plist with real paths
#      into ~/Library/LaunchAgents/
#   3. bootstraps the agent via launchctl
#   4. Kicks off an immediate run so you see status within seconds
#
# Idempotent: safe to re-run (bootout then bootstrap).
#
set -euo pipefail

REPO="${SOCCER_REPO:-$HOME/Downloads/soccer-video-pipeline}"
PLIST_SRC="$REPO/infra/com.soccer.runpoller.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.soccer.runpoller.plist"
STATE_DIR="$HOME/soccer-runs/state"
LOGS_DIR="$HOME/soccer-runs/logs"
LABEL="com.soccer.runpoller"

if [[ ! -f "$PLIST_SRC" ]]; then
  echo "error: $PLIST_SRC not found; run this from a clean checkout" >&2
  exit 1
fi

mkdir -p "$STATE_DIR" "$LOGS_DIR" "$(dirname "$PLIST_DST")"
chmod +x "$REPO/scripts/poll_current_run.sh"

# Render the template with absolute paths. macOS sed needs -i ''.
sed \
  -e "s|__REPO__|$REPO|g" \
  -e "s|__HOME__|$HOME|g" \
  "$PLIST_SRC" > "$PLIST_DST"

echo "installed plist -> $PLIST_DST"

# Bootstrap (or re-bootstrap) the agent under the current GUI user.
UID_NUM=$(id -u)
DOMAIN="gui/$UID_NUM"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  echo "bootout existing $LABEL..."
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
fi

echo "bootstrap $LABEL..."
launchctl bootstrap "$DOMAIN" "$PLIST_DST"
launchctl enable "$DOMAIN/$LABEL"
launchctl kickstart -k "$DOMAIN/$LABEL" || true

sleep 2
echo
echo "── status ────────────────────────────────────────────────"
launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state|last exit|program|PID" || true
echo
echo "── heartbeat ──────────────────────────────────────────────"
if [[ -f "$STATE_DIR/poller_last_run" ]]; then
  echo "last run: $(cat "$STATE_DIR/poller_last_run")"
else
  echo "(no heartbeat yet; first tick pending)"
fi
echo
echo "── tail of poller.log ─────────────────────────────────────"
tail -n 20 "$LOGS_DIR/poller.log" 2>/dev/null || echo "(no log yet)"
echo
echo "done. The poller will run every 5 minutes. Manual run:"
echo "  $REPO/scripts/poll_current_run.sh"
