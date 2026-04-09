# state/ — Durable run tracking

This directory holds the **committed, immutable** manifest for the currently
active detection run. It is the source of truth that survives Claude Code
session compactions, CLI restarts, and Mac reboots.

## Files

- `current_run.json` — the active run manifest (run number, job id, commit
  sha, baseline to beat, evaluation command, targets). Updated **manually**
  whenever a new run is submitted, then committed and pushed.
- `runs/` — placeholder directory; per-run result snapshots are written here
  by tooling when copied back from the Mac poller (not auto-committed).

## Live status (NOT in this repo)

The Mac-side poller (`scripts/poll_current_run.sh`, installed via launchd)
writes live status and final eval results to **`~/soccer-runs/state/`** on
the Mac:

- `run_<N>_status.json` — snapshot of the job status at the last poll tick
- `run_<N>_result.json` — final evaluation results (written once, when the
  job transitions to `complete`)
- `poller_last_run` — heartbeat timestamp (proves the poller is alive)

These files are intentionally **not** in git — they are host-local runtime
state. If you need a copy back on the dev box, scp them:

```bash
scp mac:soccer-runs/state/run_8_result.json state/runs/
```

## Recovery on a new Claude session

When a Claude session starts (fresh or post-compaction), the very first
actions for run-tracking work should be:

```bash
cat state/current_run.json                         # run config
ssh mac 'cat ~/soccer-runs/state/run_8_status.json'   # live status
ssh mac 'cat ~/soccer-runs/state/run_8_result.json'   # result if done
ssh mac 'cat ~/soccer-runs/state/poller_last_run'     # poller heartbeat
```

If the result file exists, the run is done and the poller already ran the
evaluation — read it directly.

## Submitting a new run

1. Make the code changes for the new run, commit.
2. Edit `state/current_run.json`: bump `run_number`, set `job_id` to the
   new submission id, update `commit_sha`, `submitted_at`, `baseline`,
   `notes`.
3. Commit the manifest change.
4. Push to main.
5. On the Mac: `cd ~/Downloads/soccer-video-pipeline && git pull`.
6. The poller picks up the new manifest on its next tick (every 5 min).
