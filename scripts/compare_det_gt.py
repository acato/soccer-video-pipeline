"""Compare detected events to GT events for offset debugging."""
import json
import sys

det_path = sys.argv[1]
gt1_path = sys.argv[2]
gt2_path = sys.argv[3]
video_offset = float(sys.argv[4]) if len(sys.argv) > 4 else 420
half2_start = float(sys.argv[5]) if len(sys.argv) > 5 else 4020
half2_game_offset = float(sys.argv[6]) if len(sys.argv) > 6 else 2700

det = [json.loads(l) for l in open(det_path)]
det.sort(key=lambda x: x["start_sec"])

print("=== First 10 detected events ===")
for d in det[:10]:
    print(f"  t={d['start_sec']:7.1f}s  {d['event_type']}")

print(f"\n=== All detected goals ({sum(1 for d in det if d['event_type'] == 'goal')}) ===")
for g in [d for d in det if d["event_type"] == "goal"]:
    print(f"  t={g['start_sec']:7.1f}s  reasoning={(g.get('reasoning') or '')[:100]}")

print("\n=== GT goals (mapped to video time with current offsets) ===")
for half_idx, path in enumerate([gt1_path, gt2_path]):
    with open(path) as f:
        data = json.load(f)["data"]
    for entry in data:
        for ev in entry.get("events", []):
            if ev.get("event_name") == "Goals Conceded":
                evtime_sec = entry["event_time"] / 1000.0
                if half_idx == 0:
                    video_time = evtime_sec + video_offset
                else:
                    video_time = (evtime_sec - half2_game_offset) + half2_start
                team = entry.get("team_name", "")[:30]
                print(f"  half={half_idx+1} game={evtime_sec:6.1f}s  video={video_time:6.1f}s  team={team}")

# For offset calibration: first and last GT events
print(f"\n=== Offset diagnostics ===")
for half_idx, path in enumerate([gt1_path, gt2_path]):
    with open(path) as f:
        data = json.load(f)["data"]
    first_ev_time = min(e["event_time"] for e in data) / 1000.0
    last_ev_time = max(e["event_time"] for e in data) / 1000.0
    if half_idx == 0:
        first_vid = first_ev_time + video_offset
        last_vid = last_ev_time + video_offset
    else:
        first_vid = (first_ev_time - half2_game_offset) + half2_start
        last_vid = (last_ev_time - half2_game_offset) + half2_start
    print(f"  Half {half_idx+1}: game={first_ev_time:.1f}→{last_ev_time:.1f}s  video={first_vid:.1f}→{last_vid:.1f}s")
print(f"  Detected range: {det[0]['start_sec']:.1f}→{det[-1]['start_sec']:.1f}s ({len(det)} events)")
