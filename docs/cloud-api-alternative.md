# Cloud-API Alternative — Running Without Local GPUs

The production pipeline calls a vLLM endpoint serving a custom-trained
Qwen3-VL-32B LoRA. That requires roughly **2× RTX 3090 / 4090 (≥ 48 GB
VRAM total)** or a single A100/H100 — i.e., several thousand dollars of
hardware, or a comparable cloud GPU rental.

For users without that hardware, the same pipeline can run end-to-end
against a hosted vision-language model API. This document shows how to
do it with **Anthropic Claude Sonnet 4.6** behind a
[LiteLLM](https://github.com/BerriAI/litellm) proxy — no detector code
changes required — and reports the measured cost and accuracy trade.

For the full installation including the local-vLLM path see
[install.md](install.md).

---

## 1. The architecture

```
┌─────────────────────┐   OpenAI-compatible    ┌──────────────────┐
│ scripts/            │   chat completion      │  LiteLLM proxy   │
│ run_dual_pass.py    │ ─────────────────────► │  on localhost    │
│                     │ ◄───────────────────── │  :4000           │
└─────────────────────┘                        └────────┬─────────┘
                                                        │ Anthropic format
                                                        ▼
                                           ┌────────────────────────┐
                                           │  Anthropic API         │
                                           │  claude-sonnet-4-6     │
                                           └────────────────────────┘
```

LiteLLM speaks the OpenAI chat-completions protocol on one side and the
Anthropic Messages API on the other. The dual_pass detector keeps thinking
it's talking to a local vLLM serving `qwen3-vl-32b`; LiteLLM rewrites the
call into Anthropic format. **Zero detector code changes**.

---

## 2. Setup

### One-time

Get an Anthropic API key from [console.anthropic.com](https://console.anthropic.com).
Add it to `infra/.env` (gitignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

Install LiteLLM into the same venv the pipeline already uses:

```bash
.venv/bin/pip install "litellm[proxy]"
```

### LiteLLM config

Save as `/tmp/litellm_config.yaml`:

```yaml
model_list:
  # Alias `qwen3-vl-32b` to Claude so the dual_pass detector — which
  # always sets `model: "qwen3-vl-32b"` in its requests — routes to
  # Claude without any code changes.
  - model_name: qwen3-vl-32b
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY
      max_tokens: 800
      temperature: 0

litellm_settings:
  num_retries: 3
  request_timeout: 120

general_settings:
  disable_spend_logs: true
```

### Start the proxy

```bash
ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY infra/.env | cut -d= -f2) \
  .venv/bin/litellm --config /tmp/litellm_config.yaml --port 4000
```

Verify with a quick health check:

```bash
curl -s -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl-32b","messages":[{"role":"user","content":"reply: ok"}],"max_tokens":5}'
```

Expected: a JSON response with `choices[0].message.content` containing "ok".

---

## 3. Run the pipeline

Point `--vllm-url` at the LiteLLM proxy instead of a local vLLM:

```bash
.venv/bin/python scripts/run_dual_pass.py \
  --video "/path/to/match.mp4" \
  --out-dir /tmp/soccer-pipeline/<job_id> \
  --vllm-url http://localhost:4000
```

> **Do NOT set `PREFILL_JSON_ARRAY=true`.** Sonnet 4.6 has extended-thinking
> mode that disallows assistant-message prefill. The stronger "JSON only"
> tail on the prompt is sufficient.

All downstream stages (kickoff ensemble, merge + tier tagging, reel build)
work the same as the local-vLLM path.

---

## 4. Cost (per full match)

Measured on game_20 (7230s / ~2h video, 723 single-pass windows at
`step_sec=10`, 5 frames per window at 1280×720):

| Item                         | Value      |
|------------------------------|-----------:|
| Wall time                    | **~50 min** |
| API calls                    | 723 + retries (732 total) |
| Estimated input tokens       | ~5.0 M     |
| Estimated output tokens      | ~72 K      |
| **Cost @ Sonnet 4.6**        | **~$15-16** |

Compared with a local 2× RTX 3090 run:

| Item                | Local vLLM           | Cloud (Sonnet 4.6) |
|---------------------|----------------------|---------------------|
| Hardware upfront    | ~$3,000-4,000 (2× 3090s + host) | $0          |
| Per-match cost      | electricity (~$0.50) | ~$15-16             |
| Wall time per match | 30-60 min            | ~50 min             |
| Setup complexity    | high (vLLM, drivers) | low (one config file) |

Crossover: at $15/match cloud vs ~$3,500 upfront for local hardware,
local pays back after ~230 matches. For occasional users (a few games
per season) the cloud path is unambiguously cheaper.

---

## 5. Accuracy trade — game_20 head-to-head

Both paths ran on the same source video and went through the same
downstream tier-tagging + reel-build pipeline. Ground truth: 9 goals,
12 saves, 37 shots. Coverage metric at ±60s tolerance.

| Metric                  | Local v11 LoRA | Cloud Sonnet 4.6 | Delta |
|-------------------------|---------------:|-----------------:|------:|
| **Goals — union recall** | **9/9 = 1.00** | 8/9 = 0.89       | -0.11 |
| Goals — union FPs       | 24             | 27               | +3    |
| Goals — union precision | 0.27           | 0.23             |       |
| **Saves — union recall** | **10/12 = 0.83** | 8/12 = 0.67    | -0.16 |
| Saves — union FPs       | 40             | 38               | -2    |
| Saves — union precision | 0.20           | 0.17             |       |
| **Shots — union recall** | **29/37 = 0.78** | 26/37 = 0.70   | -0.08 |
| Shots — union FPs       | 12             | 19               | +7    |
| Shots — union precision | 0.71           | 0.58             |       |

Local wins on every recall axis. The LoRA is specifically fine-tuned on
amateur-soccer event clips, so this isn't surprising — the gap reflects
real training signal that the off-the-shelf Sonnet doesn't have.

Where Sonnet diverges most:

- **Far more raw goal candidates** (14 vs 1) — Sonnet over-emits "goal"
  on near-misses without seeing the celebration/restart that the LoRA
  was trained to require.
- **Many more throw-ins** (70 vs 23) — Sonnet aggressively flags any
  player near a sideline.
- **No corner kicks detected** — Sonnet doesn't recognize the corner
  signature at sideline-camera amateur distance, while the LoRA was
  trained on examples.
- **Almost no free_kick_shot** (1 vs 23) — Sonnet's prompt requires a
  visible defensive wall, which is rare in amateur footage.

The differing distribution affects the tier pipeline downstream: more
goal candidates means more FPs to chase; missing corner detections
means the inferred-save tier loses some recall.

---

## 6. When to use which

**Use the local v11 LoRA path** if:
- You're processing more than ~250 matches total
- You have 2× modern GPUs already
- Hitting the recall floor matters (high-recall reels for coaches)

**Use the cloud Sonnet path** if:
- You don't have GPUs and don't want to buy them
- You're processing tens of matches, not hundreds
- You're OK trading ~0.1 recall for ~$15/match

**Try other cloud models via LiteLLM** if Sonnet's accuracy isn't
acceptable: GPT-4o, Gemini 2.5 Flash, etc. all work through the same
proxy — change one line in `litellm_config.yaml`.

---

## 7. Tradeoffs and limits documented elsewhere

- The `dual_pass_detector.py` prompt has an explicit "JSON only" tail.
  This was added when bringing Sonnet on board (its default style is
  to wrap responses in markdown analysis); the v11 LoRA ignores it.
- Sonnet 4.6 doesn't allow assistant-message prefill (extended thinking
  conflict). Older Sonnet variants (`claude-sonnet-4-5-20250929`) do.
  See `PREFILL_JSON_ARRAY` in `dual_pass_detector.py` if you switch.
- LiteLLM's `disable_spend_logs: true` means no built-in cost tracking.
  For per-match cost accounting, query the Anthropic admin API after
  the run, or do the back-of-envelope from window count.
