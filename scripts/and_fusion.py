"""AND-fusion: keep ensemble dets within ±90s of a dual_pass detection."""
import json
from pathlib import Path

TOL = 90.0


def aggregate_relaxed(labels):
    labs = sorted(labels, key=lambda x: x[0])
    if any(l == "celebration" for _, l in labs):
        return "GOAL"
    for i, (_, l) in enumerate(labs):
        if l == "goal":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
    for i, (_, l) in enumerate(labs):
        if l == "kickoff_restart":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
            for _, before in labs[:i]:
                if before in ("goal", "celebration", "set_piece"):
                    return "GOAL"
    return "NO"


GAMES = {
    "game_20": (["/tmp/kickoff_game20_1H_formation_base.jsonl",
                 "/tmp/kickoff_game20_2H_formation_base.jsonl"],
                None,
                "/tmp/soccer-pipeline/a0f8f93c-8611-466b-934c-8cd48a2aee00/events.jsonl",
                [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
                [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]]),
    "game_22": (["/tmp/kickoff_game_22_formation_v2_base.jsonl"],
                None,
                "/tmp/soccer-pipeline/031ee71a-7a4f-4605-a642-9bff003e4804/events.jsonl",
                [g + 195 for g in [1559.7, 2195.4]] +
                [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]]),
    "game_21": (["/tmp/kickoff_game_21_formation_v2_base.jsonl"],
                None,
                "/tmp/soccer-pipeline/60ed91b3-ebcb-4ddb-ac18-768923a17419/events.jsonl",
                [g + 65 for g in [1578.6, 2008.6]]),
    "rush": (["/tmp/kickoff_rush_formation_v2_base.jsonl"],
             "/tmp/kickoff_rush_pattern_v11_0191.jsonl",
             "/tmp/soccer-pipeline/47d1358c-268c-4577-b4e1-f9943f51be6a/events.jsonl",
             [g + 418 for g in [383.1, 647.0]] +
             [g + 418 + 770 for g in [3038.4, 3263.5]]),
}


def load_ensemble(form_paths, pat_path):
    out = []
    for fp in form_paths:
        p = Path(fp)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if aggregate_relaxed(r.get("_vlm_labels", [])) == "GOAL":
                out.append(r["start_sec"])
    if pat_path and Path(pat_path).exists():
        for line in Path(pat_path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_vlm_verdict") == "GOAL":
                out.append(r["start_sec"])
    return sorted(out)


def load_dual_pass_goals(events_path):
    p = Path(events_path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event_type") == "goal":
            out.append(r.get("timestamp_start", r.get("start_sec")))
    return sorted(out)


def dedup60(ts):
    out = []
    for t in sorted(ts):
        if out and (t - out[-1]) <= 60:
            continue
        out.append(t)
    return out


def score(dets, gts, tol=TOL):
    used = set()
    tp = 0
    for d in sorted(dets):
        for i, g in enumerate(gts):
            if i in used: continue
            if abs(d - g) <= tol:
                used.add(i); tp += 1; break
    fp = len(dets) - tp
    fn = len(gts) - tp
    return tp, fp, fn


print(f"{'pipeline':<28} {'TP':>3} {'FP':>4} {'FN':>3} {'recall':>6} {'prec':>6} {'F1':>5}")
print("-" * 65)

for fusion_mode in ("ensemble_alone", "dual_pass_alone", "AND", "OR", "OR_dedup60"):
    total_tp = 0; total_fp = 0; total_fn = 0
    for game, (form_paths, pat_path, dp_path, gts) in GAMES.items():
        ens = load_ensemble(form_paths, pat_path)
        ens = dedup60(ens)  # baseline 60s dedup
        dp = load_dual_pass_goals(dp_path)
        if fusion_mode == "ensemble_alone":
            dets = ens
        elif fusion_mode == "dual_pass_alone":
            dets = dp
        elif fusion_mode == "AND":
            dets = [e for e in ens if any(abs(e - d) <= TOL for d in dp)]
        elif fusion_mode == "OR":
            dets = sorted(set(ens) | set(dp))
        elif fusion_mode == "OR_dedup60":
            dets = dedup60(sorted(set(ens) | set(dp)))
        tp, fp, fn = score(dets, gts)
        total_tp += tp; total_fp += fp; total_fn += fn
    rec = total_tp / max(1, total_tp + total_fn)
    prec = total_tp / max(1, total_tp + total_fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(f"{fusion_mode:<28} {total_tp:>3} {total_fp:>4} {total_fn:>3} "
          f"{rec:>6.2f} {prec:>6.2f} {f1:>5.2f}")
