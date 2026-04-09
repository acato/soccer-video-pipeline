#!/usr/bin/env bash
#
# start_poller_loop.sh — run poll_current_run.sh on a loop, detached from
# the terminal, in an SSH-inherited process context that has Full Disk
# Access on macOS (unlike launchd agents in ~/Downloads).
#
# Commands:
#   bash scripts/start_poller_loop.sh            # start (idempotent)
#   bash scripts/start_poller_loop.sh status     # show pid + last ticks
#   bash scripts/start_poller_loop.sh stop       # kill the loop
#   bash scripts/start_poller_loop.sh restart    # stop then start
#
# Environment:
#   SOCCER_REPO        default: $HOME/Downloads/soccer-video-pipeline
#   SOCCER_STATE_DIR   default: $HOME/soccer-runs/state
#   SOCCER_LOGS_DIR    default: $HOME/soccer-runs/logs
#   POLL_INTERVAL      default: 300 (seconds)
#
set -u

REPO="${SOCCER_REPO:-$HOME/Downloads/soccer-video-pipeline}"
STATE_DIR="${SOCCER_STATE_DIR:-$HOME/soccer-runs/state}"
LOGS_DIR="${SOCCER_LOGS_DIR:-$HOME/soccer-runs/logs}"
INTERVAL="${POLL_INTERVAL:-300}"

PID_FILE="$STATE_DIR/poller_loop.pid"
LOOP_LOG="$LOGS_DIR/poller_loop.log"
POLLER="$REPO/scripts/poll_current_run.sh"

mkdir -p "$STATE_DIR" "$LOGS_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

cmd_status() {
  if is_running; then
    local pid
    pid=$(cat "$PID_FILE")
    echo "running pid=$pid interval=${INTERVAL}s"
    if [[ -f "$STATE_DIR/poller_last_run" ]]; then
      echo "last tick: $(cat "$STATE_DIR/poller_last_run")"
    fi
    echo
    echo "── loop log tail ──"
    tail -n 8 "$LOOP_LOG" 2>/dev/null || true
    echo
    echo "── poller log tail ──"
    tail -n 12 "$LOGS_DIR/poller.log" 2>/dev/null || true
  else
    echo "not running"
    [[ -f "$PID_FILE" ]] && echo "(stale pid file: $(cat "$PID_FILE"))"
    return 1
  fi
}

cmd_stop() {
  if ! is_running; then
    echo "not running"
    rm -f "$PID_FILE"
    return 0
  fi
  local pid
  pid=$(cat "$PID_FILE")
  echo "stopping pid=$pid"
  kill "$pid" 2>/dev/null || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "stopped"
}

cmd_start() {
  if is_running; then
    echo "already running pid=$(cat "$PID_FILE")"
    cmd_status
    return 0
  fi

  [[ -x "$POLLER" ]] || chmod +x "$POLLER"
  if [[ ! -f "$POLLER" ]]; then
    echo "error: $POLLER not found"
    return 1
  fi

  echo "starting loop (interval=${INTERVAL}s)"
  # Double-fork via setsid-free disown so the loop survives SSH exit.
  nohup bash -c "
    echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] loop-start pid=\$\$ interval=$INTERVAL\" >> '$LOOP_LOG'
    trap 'echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] loop-stop pid=\$\$\" >> \"$LOOP_LOG\"; rm -f \"$PID_FILE\"; exit 0' TERM INT
    while true; do
      bash '$POLLER' >> '$LOOP_LOG' 2>&1 || echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] poller-exit=\$?\" >> '$LOOP_LOG'
      sleep $INTERVAL
    done
  " >/dev/null 2>&1 &
  local pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid" > "$PID_FILE"
  sleep 1
  if is_running; then
    echo "started pid=$pid"
    cmd_status
  else
    echo "failed to start — check $LOOP_LOG"
    return 1
  fi
}

cmd_restart() {
  cmd_stop || true
  sleep 1
  cmd_start
}

case "${1:-start}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  restart) cmd_restart ;;
  *)       echo "usage: $0 {start|stop|status|restart}"; exit 2 ;;
esac
