import json
cands = [json.loads(l) for l in open("/tmp/kickoff_game_21_formation_v2.jsonl")]
print(f"game_21 v2 candidates in [1500, 1800]:")
for c in cands:
    if 1500 <= c["start_sec"] <= 1800:
        cs = c.get("_cluster_start", "?")
        ce = c.get("_cluster_end", "?")
        size = c.get("_cluster_size", "?")
        print(f"  t={c['start_sec']} size={size} span={cs}-{ce}")
