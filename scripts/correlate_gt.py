"""Correlate detected events with ground truth JSON from match analytics."""
import json
import sys

VIDEO_OFFSET = 418.0  # Game starts at 6:58 in video
MATCH_WINDOW = 60.0   # seconds tolerance for matching

GT_FILES = [
    "/Volumes/transit/08 GA (U19) vs Washington Rush U19 (W)_1st Half.json",
    "/Volumes/transit/08 GA (U19) vs Washington Rush U19 (W)_2nd Half.json",
]

def load_gt():
    gt_events = []
    for path in GT_FILES:
        with open(path) as f:
            data = json.load(f)["data"]
        for entry in data:
            team = "Rush" if "Rush" in entry.get("team_name", "") else "GA"
            player = entry.get("player_name", "")
            t_sec = entry["event_time"] / 1000.0
            for ev in entry.get("events", []):
                name = ev["event_name"]
                prop = ev.get("property", {})
                typ = prop.get("Type", "")
                outcome = prop.get("Outcome", "")
                mapped = None
                if name == "Shots & Goals" and outcome == "Goals": mapped = "goal"
                elif name == "Shots & Goals" and outcome == "Shots On Target": mapped = "shot_on_target"
                elif name == "Shots & Goals" and outcome == "Shots Off Target": mapped = "shot_off_target"
                elif name == "Shots & Goals" and outcome == "Blocked Shots": mapped = "blocked_shot"
                elif name == "Saves" and typ == "Catches": mapped = "catch"
                elif name == "Saves" and typ == "Parries": mapped = "parry"
                elif name == "Set Pieces" and typ == "Goal Kicks": mapped = "goal_kick"
                elif "Corners" in typ: mapped = "corner_kick"
                elif name == "Set Pieces" and typ == "Throw-Ins": mapped = "throw_in"
                elif name == "Set Pieces" and "Freekick" in typ: mapped = "free_kick"
                elif name == "Set Pieces" and "Penalty" in typ: mapped = "penalty"
                if mapped:
                    gt_events.append({"type": mapped, "gt_sec": t_sec, "team": team, "player": player})
    gt_events.sort(key=lambda x: x["gt_sec"])
    return gt_events


def load_detected(events_jsonl):
    detected = []
    with open(events_jsonl) as f:
        for line in f:
            e = json.loads(line)
            detected.append({"type": e["event_type"], "gt_sec": e["timestamp_start"] - VIDEO_OFFSET})
    detected.sort(key=lambda x: x["gt_sec"])
    return detected


def fmt(s):
    mm = int(abs(s) // 60)
    ss = abs(s) % 60
    return "%s%02d:%04.1f" % ("-" if s < 0 else "", mm, ss)


def correlate(gt_events, detected):
    categories = [
        ("GOALS",               ["goal"],        ["goal"]),
        ("SAVES - Elena catch", ["catch"],        ["catch"],         "Elena"),
        ("SAVES - Elena parry", ["parry"],        ["shot_stop_diving"], "Elena"),
        ("CORNERS",             ["corner_kick"],  ["corner_kick"]),
        ("GOAL KICKS",          ["goal_kick"],    ["goal_kick"]),
        ("FREE KICKS",          ["free_kick"],    ["free_kick_shot"]),
        ("THROW-INS",           ["throw_in"],     ["throw_in"]),
        ("SHOTS ON TARGET",     ["shot_on_target"], ["shot_on_target"]),
        ("SHOTS OFF TARGET",    ["shot_off_target"], ["shot_off_target"]),
        ("PENALTY",             ["penalty"],      ["penalty"]),
    ]

    for row in categories:
        label, gt_types, det_types = row[0], row[1], row[2]
        player_filter = row[3] if len(row) > 3 else None

        gt_sub = [g for g in gt_events if g["type"] in gt_types]
        if player_filter:
            gt_sub = [g for g in gt_sub if player_filter in g.get("player", "")]
        det_sub = [d for d in detected if d["type"] in det_types]

        print("=== %s === (GT=%d  Det=%d)" % (label, len(gt_sub), len(det_sub)))

        det_used = set()
        tp, fn_list = 0, []
        for g in gt_sub:
            best_i, best_d = None, 999999
            for i, d in enumerate(det_sub):
                if i in det_used:
                    continue
                dist = abs(g["gt_sec"] - d["gt_sec"])
                if dist < best_d:
                    best_d = dist
                    best_i = i
            if best_i is not None and best_d <= MATCH_WINDOW:
                det_used.add(best_i)
                tp += 1
                d = det_sub[best_i]
                print("  TP  GT=%s  Det=%s  delta=%+.0fs  %s %s" % (
                    fmt(g["gt_sec"]), fmt(d["gt_sec"]), d["gt_sec"] - g["gt_sec"],
                    g.get("team", ""), g.get("player", "")))
            else:
                fn_list.append(g)
                print("  FN  GT=%s  (missed)  %s %s" % (
                    fmt(g["gt_sec"]), g.get("team", ""), g.get("player", "")))

        fp_list = [det_sub[i] for i in range(len(det_sub)) if i not in det_used]
        for d in fp_list:
            print("  FP  Det=%s  (spurious)" % fmt(d["gt_sec"]))

        prec = 100 * tp / (tp + len(fp_list)) if (tp + len(fp_list)) > 0 else 0
        rec = 100 * tp / (tp + len(fn_list)) if (tp + len(fn_list)) > 0 else 0
        print("  >> TP=%d  FN=%d  FP=%d  Precision=%.0f%%  Recall=%.0f%%" % (
            tp, len(fn_list), len(fp_list), prec, rec))
        print()


if __name__ == "__main__":
    events_jsonl = sys.argv[1] if len(sys.argv) > 1 else \
        "/tmp/soccer-pipeline/b558edb3-4be9-4085-84be-ed036c7cb18f/events.jsonl"
    gt = load_gt()
    det = load_detected(events_jsonl)
    correlate(gt, det)
