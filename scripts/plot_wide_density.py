"""Plot rolling wide_shot density per game to see if halftime is visible."""
import json
import sys

FILES = {
    "game_22": "/tmp/kickoff_game_22_frames.jsonl",
    "game_21": "/tmp/kickoff_game_21_frames.jsonl",
    "rush":    "/tmp/kickoff_rush_frames.jsonl",
}

WINDOW = 30  # frames (= 150s at 5s interval)


def density_profile(frames, signal_fn, window):
    out = []
    n = len(frames)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        v = sum(1 for k in range(lo, hi) if signal_fn(frames[k]))
        out.append((frames[i]["t"], v / max(1, hi - lo)))
    return out


for label, path in FILES.items():
    frames = [json.loads(l) for l in open(path)]
    wide = density_profile(frames, lambda f: f.get("wide_shot"), WINDOW)
    ball = density_profile(frames, lambda f: f.get("ball") is not None, WINDOW)

    print(f"\n=== {label} ({len(frames)} frames over {frames[0]['t']:.0f}-{frames[-1]['t']:.0f}s) ===")
    # ASCII profile every 150s
    print(f"  {'t(s)':>6}  {'wide%':>6}  {'ball%':>6}  bar (wide: # = >50%)")
    step = 30  # every 30 frames = 150s
    for i in range(0, len(frames), step):
        t, w = wide[i]
        _, b = ball[i]
        bar = "#" * int(w * 30)
        flag = ""
        if w < 0.2: flag = "  <- NON-GAME"
        elif w > 0.7: flag = "  <- GAME"
        print(f"  {t:>6.0f}  {w*100:>5.0f}%  {b*100:>5.0f}%  {bar}{flag}")
