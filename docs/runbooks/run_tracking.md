# Run tracking & session resilience

## What this is

A durable system for tracking long-running detection jobs (typically 1.5–4 hours)
across Claude Code session compactions, CLI restarts, SSH disconnects, and
other ephemeral-state losses. It was built after losing track of Run #8
mid-flight when the Claude session compacted, the in-session cron loop
evaporated, and I spent 20 minutes curling the wrong machine on the wrong
port trying to figure out what happened.

The core principle: **Claude-side state is ephemeral. Push everything
important onto disk or into a Mac-side process that sshd spawns.**

## Topology

```
┌──────────────────┐            ┌───────────────────────────┐
│  Windows dev box │            │       Mac Studio          │
│  (Claude Code)   │   ssh mac  │       ssh: mac            │
│                  │──────────▶ │   10.10.7.166             │
│  Edits repo,     │            │                           │
│  commits to      │            │  ┌──────────────────────┐ │
│  main, pushes    │            │  │ soccer-video-pipeline│ │
└──────────────────┘            │  │  (Downloads/…)       │ │
         │                      │  │  API :8088           │ │
         │ pushes               │  │  Celery worker       │ │
         ▼                      │  │  infra-redis-1       │ │
   ┌────────────┐    git pull   │  └──────────────────────┘ │
   │  GitHub    │◀──────────────│                           │
   │  main      │               │  ┌──────────────────────┐ │
   └────────────┘               │  │ nohup poller loop    │ │
                                │  │  ppid=1 (detached)   │ │
                                │  │  tick every 300s     │ │
                                │  └──────────┬───────────┘ │
                                │             │             │
                                │             ▼             │
                                │  ~/soccer-runs/state/*    │
                                │  ~/soccer-runs/logs/*     │
                                └───────────────────────────┘
                                              │
                                              │ HTTP POST
                                              ▼
                                 ┌───────────────────────────┐
                                 │  LLM server 10.10.2.222   │
                                 │  vLLM (Qwen3-VL-8B + 32B) │
                                 └───────────────────────────┘
```

## The two layers of durable state

### Layer 1 — `state/current_run.json` (committed, immutable per run)

Source of truth for which run is active. Written manually when submitting
a new run, committed to main, pulled on the Mac. Schema:

```json
{
  "$schema_version": 1,
  "run_number": 8,
  "run_label": "run_8",
  "job_id": "0eb76694-edfa-46d3-9ba0-d2102d781cac",
  "commit_sha": "fd6907f",
  "submitted_at": "2026-04-09T14:38:15Z",
  "host": "mac",
  "api_url": "http://127.0.0.1:8088",
  "events_file_template": "/tmp/soccer-pipeline/{job_id}/diagnostics/dual_pass_events.jsonl",
  "evaluate_script": "scripts/evaluate_detection.py",
  "evaluate_args": ["--tolerance", "30"],
  "baseline": {"run": "7c", "f1": 0.28, "notes": "…"},
  "targets": {"f1": 0.35, "goal_kick_tp_min": 8, …},
  "notes": "Run #8 - corner_kick description strengthened, …"
}
```

Fields explained:

| Field | Purpose |
|---|---|
| `run_number` / `run_label` | Human identifiers; label is used as file prefix in `~/soccer-runs/` |
| `job_id` | Celery/API job id for status polling and events-file lookup |
| `commit_sha` | Which commit the run is built from (for blame when it regresses) |
| `submitted_at` | ISO8601 UTC — used to spot stuck jobs |
| `api_url` | Where the poller queries job status |
| `events_file_template` | Where the dual-pass detector writes its final jsonl |
| `evaluate_script` / `evaluate_args` | What the poller runs on completion; args flow through so we can bump `--tolerance` without touching code |
| `baseline` | What F1 to beat; used to compute `delta_f1_vs_baseline` in the result file |
| `targets` | Aspirational per-type goals; currently informational |
| `notes` | Free text for what this run is trying to prove |

Only the active run is in `current_run.json`. Historical runs live in
`memory/run_metrics_trail.md` (and the result files from Layer 2 are
archived to `state/runs/` when needed).

### Layer 2 — `~/soccer-runs/` on the Mac (poller runtime state)

Not in git. Written by the nohup loop. Survives any Claude or SSH event.

```
~/soccer-runs/
├── state/
│   ├── current_run.json          # (optional: copy of Layer 1, not used)
│   ├── poller_loop.pid           # nohup loop PID
│   ├── poller_last_run           # heartbeat, updated every tick
│   ├── run_<N>_status.json       # latest API snapshot (overwritten each tick)
│   ├── run_<N>_eval.json         # raw --json-out from evaluate_detection.py
│   └── run_<N>_result.json       # final enriched result (written once, triggers "done")
└── logs/
    ├── poller.log                # per-tick log (rotated at ~2 MB)
    ├── poller.log.1              # rotated
    ├── poller_loop.log           # loop-level events (start/stop/poller exit codes)
    └── run_<N>_eval.log          # stdout+stderr from evaluate_detection.py
```

**`run_<N>_result.json` is the "done" sentinel.** Its presence tells the
poller to stop re-polling and re-evaluating. The poller short-circuits
at the top of every tick if this file exists.

## The scripts

### `scripts/poll_current_run.sh` — single-tick poller
Stateless. Idempotent. Called once per tick by the loop wrapper.

What it does:
1. Write heartbeat file (always, before anything else).
2. Locate a working Python interpreter.
3. Read `state/current_run.json` from the repo checkout.
4. Short-circuit if `run_<N>_result.json` already exists.
5. `curl $api_url/jobs/$job_id` with a 15s timeout.
6. Write the status snapshot atomically (temp file + rename).
7. Dispatch on status:
   - `complete` → run the evaluator with `--json-out`, then assemble the
     enriched result file including baseline delta and send a macOS
     notification.
   - `failed`/`error` → write a failure result file and notify.
   - otherwise (`detecting`/`segmenting`/…) → do nothing more.
8. All writes use `.tmp` + `os.replace()` so a crash mid-write never
   leaves a half-written file.

Environment variable overrides (for testing or alternate layouts):
- `SOCCER_REPO`  — default `$HOME/Downloads/soccer-video-pipeline`
- `SOCCER_STATE_DIR` — default `$HOME/soccer-runs/state`
- `SOCCER_LOGS_DIR`  — default `$HOME/soccer-runs/logs`

Exit code is always 0 so launchd/cron don't throttle it; errors are logged.

### `scripts/start_poller_loop.sh` — detached loop wrapper
`start | stop | status | restart` interface over a nohup'd `while true`
around the single-tick poller.

```bash
bash scripts/start_poller_loop.sh start     # starts, writes PID file, detaches
bash scripts/start_poller_loop.sh status    # shows PID + last tick + log tails
bash scripts/start_poller_loop.sh stop      # TERM, then KILL if still alive
bash scripts/start_poller_loop.sh restart   # stop; sleep 1; start
```

The started process is `nohup bash -c 'while…; done' &` with `disown`,
which detaches the job from the shell so the process survives SSH exit.
You can confirm detachment via `ps -p <pid> -o ppid=` — it should be `1`
(init) once the SSH parent exits, not the sshd PID.

Interval defaults to 300s, overridable via `POLL_INTERVAL`.

Handles stale PID files (process gone but file still there) by treating
them as not-running and letting `start` proceed.

### `scripts/evaluate_detection.py --json-out PATH`
Added a flag that dumps per-type and overall metrics to a JSON file after
the existing stdout report. Schema:

```json
{
  "events_file": "…",
  "tolerance_sec": 30,
  "gt_total": 156,
  "gt_counts": {"goal_kick": 23, …},
  "detected_total": 147,
  "detected_counts": {"throw_in": 28, …},
  "per_type": {
    "catch": {"gt": 12, "detected": 5, "tp": 1, "fn": 11, "fp": 4,
              "precision": 0.20, "recall": 0.083, "f1": 0.118},
    …
  },
  "overall": {"tp": 36, "fn": 120, "fp": 111,
              "precision": 0.245, "recall": 0.231, "f1": 0.238}
}
```

This is what the poller reads to build `run_<N>_result.json`. No more
stdout grepping. Backward compatible — omit the flag and behavior is
unchanged.

### `scripts/install_poller_launchd.sh` + `infra/com.soccer.runpoller.plist`
**Currently unused.** Kept in the tree for the day we migrate the repo
out of `~/Downloads/`. See "Why not launchd" below.

## Why not launchd? (The TCC gotcha)

macOS TCC (Transparency, Consent, and Control) protects `~/Downloads/`,
`~/Documents/`, `~/Desktop/` as user-sensitive directories. Processes
without Full Disk Access cannot read from them.

- **launchd user agents have no FDA by default.** Installing the plist
  at `~/Library/LaunchAgents/com.soccer.runpoller.plist` and bootstrapping
  it produces:
  ```
  bash: /Users/aless/Downloads/soccer-video-pipeline/scripts/poll_current_run.sh:
         Operation not permitted
  ```
  `launchctl print` shows `last exit code = 126`.
- **`cron` has the same problem** — same sandbox, same denial.
- **`sshd` inherits FDA**, so `ssh mac 'bash …'` works against files
  under `~/Downloads/`. Any process spawned by an SSH session inherits
  that access, including `nohup`'d children that survive the session.

The workaround is to run the loop in an SSH-originated process tree.
That's exactly what `start_poller_loop.sh` does.

### Alternatives considered
1. **Move the repo out of `~/Downloads/`** — the permanent fix. Would
   require recreating the Python venv (shebangs hardcode absolute paths)
   and updating the `/tmp/start-pipeline.sh` launcher plus any symlinks.
   Not done yet because the pipeline is actively running and the loop
   workaround is sufficient. Queue it for the next maintenance window.
2. **Grant `/bin/bash` Full Disk Access via System Settings** — one GUI
   click. Rejected because it's global (affects every shell script on
   the system) and requires an interactive step I can't automate.
3. **Copy scripts + state out of `~/Downloads/`** — rejected because the
   poller still needs to execute `scripts/evaluate_detection.py` from the
   repo and import the venv Python, both of which would be TCC-denied.
4. **tmux/screen** — tmux not installed; `screen` is available but adds
   a dependency for no benefit over plain nohup.

## Failure modes and their handling

| Failure | What breaks | How this system survives |
|---|---|---|
| Claude context compaction | In-session cron jobs, TodoWrite, agent memory | Manifest file + poller loop are both outside Claude |
| Claude CLI restart | Same | Same |
| SSH session closes | ppid transitions from sshd to init | `nohup` + `disown` detach the loop before the parent exits |
| Laptop running Claude goes to sleep | Nothing relevant | Poller keeps running on the Mac |
| Mac reboot | The loop dies | **Not handled** — re-run `start_poller_loop.sh start` after reboot. In 31+ days of Mac uptime this has not happened yet. |
| API crash on Mac | curl returns non-200 | Status file records `status: unreachable`; next tick retries |
| Worker killed mid-job | Job state stuck at `detecting` | Poller keeps polling; heartbeat stays fresh so you can see it's alive but the job isn't progressing. Result file is never written. |
| LLM server crash | Worker stalls on 32B calls | Same as above |
| Events file missing on completion | `dual_pass_events.jsonl` not where expected | Poller writes an error result file with `status: events_file_missing` and stops polling |
| evaluate_detection.py crashes | Non-zero exit from evaluator | Poller writes `status: eval_failed` result with a pointer to `run_<N>_eval.log` and stops polling. Restart by deleting the result file. |
| Half-written status/result file | — | All writes use `.tmp` + `os.replace()`. Readers only ever see complete files. |
| Log bloat | poller.log grows unbounded | Rotated to `poller.log.1` at ~2 MB at the top of each tick |
| Multiple concurrent runs submitted | Manifest holds one run only | Not supported by design; submit one at a time |

## Usage — day-to-day

### Session start protocol (every new Claude session)
```bash
# 1. Which run is active?
cat state/current_run.json

# 2. Is it done yet? (N = run_number from step 1)
ssh mac 'cat ~/soccer-runs/state/run_<N>_result.json'
# If the file exists the run is done and already evaluated — read it.
# If "No such file": run is still in flight, check status instead:
ssh mac 'cat ~/soccer-runs/state/run_<N>_status.json'

# 3. Is the poller alive?
ssh mac 'cat ~/soccer-runs/state/poller_last_run'
# Should be < 10 minutes old. If stale:
ssh mac 'cd ~/Downloads/soccer-video-pipeline && bash scripts/start_poller_loop.sh restart'
```

### Submitting a new run
```bash
# 1. Code changes for the new run
git add …
git commit -m "fix(triage): Run #9 …"

# 2. Submit the job (existing pipeline_cli.py or direct curl)
curl -X POST http://127.0.0.1:8088/jobs -d @new_run.json
# → returns {"job_id": "abc-123-…"}

# 3. Update the manifest
# Edit state/current_run.json: bump run_number, paste new job_id,
# update commit_sha (the one you just committed), submitted_at,
# baseline (to whatever the current best run is), notes.

git add state/current_run.json
git commit -m "state: track Run #9"
git push

# 4. Pull on the Mac
ssh mac 'cd ~/Downloads/soccer-video-pipeline && git pull'

# 5. Optional — kick the loop so it picks up the new manifest immediately
#    instead of waiting for the next tick
ssh mac 'cd ~/Downloads/soccer-video-pipeline && bash scripts/poll_current_run.sh'
```

The loop will automatically run the evaluator when the new job completes
and write `run_<N>_result.json`. You'll get a macOS notification at that
moment too (`osascript display notification`).

### After a Mac reboot
```bash
ssh mac 'cd ~/Downloads/soccer-video-pipeline && bash scripts/start_poller_loop.sh start'
```

That's it. The loop reads the current manifest and picks up wherever
things stand.

### Debugging a "stuck" run
```bash
# Is the poller alive?
ssh mac 'bash ~/Downloads/soccer-video-pipeline/scripts/start_poller_loop.sh status'

# What has the poller been seeing?
ssh mac 'tail -30 ~/soccer-runs/logs/poller.log'

# What does the API think?
ssh mac 'curl -s http://127.0.0.1:8088/jobs/<job_id>' | jq .status,.progress_pct

# Are the worker + redis + LLM server all up?
ssh mac 'ps -ef | grep -E "celery|uvicorn" | grep -v grep'
ssh mac 'docker ps --format "{{.Names}} {{.Status}}"'
ssh llm 'pgrep -f vllm && curl -s http://10.10.2.222:8000/v1/models | jq .data[0].id'
```

### Force re-evaluation after a script fix
```bash
ssh mac 'rm ~/soccer-runs/state/run_<N>_result.json'
ssh mac 'bash ~/Downloads/soccer-video-pipeline/scripts/poll_current_run.sh'
# Reads the still-complete job status, runs the evaluator again,
# writes a fresh result file.
```

## What does NOT belong in this system

- **In-session cron jobs (`CronCreate`)** for tracking the active run.
  They die on compaction. Use them only for in-session convenience
  polling that you're OK losing.
- **TodoWrite entries** as the authoritative list of which runs are
  pending. Same reason.
- **Agent memory / variables** for the job id or baseline F1. Same.
- **Claude's `scheduled-tasks` MCP tool** — subject to the same session
  lifetime as `CronCreate`; verified to not survive compaction.

It is *fine* to use all of these as **mirrors** of the durable state for
convenience within one session. Just never make them authoritative.

## File-by-file inventory

Committed to main:
```
state/current_run.json              Layer 1 manifest
state/README.md                     Quick reference for humans
state/runs/.gitkeep                 Placeholder for scp'd result archives
scripts/poll_current_run.sh         Single-tick poller (idempotent)
scripts/start_poller_loop.sh        nohup loop wrapper
scripts/install_poller_launchd.sh   (unused, kept for repo migration)
infra/com.soccer.runpoller.plist    (unused, kept for repo migration)
scripts/evaluate_detection.py       +--json-out flag
.gitattributes                      Force LF on sh/plist/json/py for Mac
docs/runbooks/run_tracking.md       This document
```

Mac-local (not in git):
```
~/soccer-runs/state/poller_loop.pid
~/soccer-runs/state/poller_last_run
~/soccer-runs/state/run_<N>_status.json
~/soccer-runs/state/run_<N>_eval.json
~/soccer-runs/state/run_<N>_result.json
~/soccer-runs/logs/poller.log
~/soccer-runs/logs/poller_loop.log
~/soccer-runs/logs/run_<N>_eval.log
```

## Historical context — why this was needed

Run #7c established F1=0.28 as a baseline on 2026-04-08. Run #8 was
submitted at 2026-04-09T14:38Z with targeted prompt changes for
corner_kick / goal_kick / throw_in / kickoff. Expected to take ~2 hours.

During Run #8:
1. Thermal throttling on the LLM server stretched the run.
2. The Claude session context compacted while waiting.
3. The in-session cron loop I had scheduled to poll the job and run the
   evaluator evaporated with the compaction.
4. The new session started with a stale memory file claiming the API was
   on port 8080 (actually 8088) and didn't know the pipeline runs on the
   Mac at all — I spent 20 minutes trying to curl localhost on the
   Windows box before realizing my mistake.
5. By the time I found the right host/port, Run #8 had finished
   (completed at ~17:06 UTC) and I nearly missed evaluating it.

The resilience system was built immediately after that recovery, and the
first thing the new poller did when installed was auto-evaluate Run #8.

The evaluation showed **F1 = 0.238, a −0.042 regression vs Run #7c**.
Per-type breakdown and Run #9 plan are in `memory/run_metrics_trail.md`.
The fact that the infrastructure caught the regression automatically —
while we were still building the infrastructure — was a good validation.

## Open items

- [ ] Migrate `~/Downloads/soccer-video-pipeline` to `~/src/soccer-video-pipeline`
      so launchd becomes a viable option (would replace the nohup loop
      with something that auto-starts at boot).
- [ ] Fix the memory file port (8088) — done in
      `memory/infra_deployment.md`, but worth a grep for any other
      lingering `8080` references.
- [ ] Optionally scp `run_<N>_result.json` back to `state/runs/` on the
      dev box after each run, and commit as a historical archive. Today
      they only live on the Mac.
- [ ] Consider a `pre-submit` helper script that updates
      `state/current_run.json` programmatically instead of by hand-edit,
      to reduce the chance of typos in the `job_id` field.
