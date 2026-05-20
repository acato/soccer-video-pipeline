# Runbook: Processing a New Match

## Automatic (Watcher) Flow
1. Drop MP4 to NAS_MOUNT_PATH
2. Watcher detects file after 30s stable size → job auto-created
3. Monitor at http://localhost:5555

## Manual Submission
```bash
# CLI (preferred)
python infra/scripts/pipeline_cli.py submit matches/game.mp4 \
  --team "Home FC" --team-outfield blue --team-gk neon_yellow \
  --opponent "Away United" --opponent-outfield red --opponent-gk neon_green

# curl fallback
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "nas_path": "matches/game.mp4",
    "match_config": {
      "team": {
        "team_name": "Home FC",
        "outfield_color": "blue",
        "gk_color": "neon_yellow"
      },
      "opponent": {
        "team_name": "Away United",
        "outfield_color": "red",
        "gk_color": "neon_green"
      }
    },
    "reel_types": ["keeper", "highlights"]
  }'
```

### match_config fields

| Field | Description | Example |
|-------|-------------|---------|
| `team.team_name` | Name of the team whose GK reel is produced | `"Home FC"` |
| `team.outfield_color` | Named color of outfield kit | `"blue"` |
| `team.gk_color` | Named color of GK kit (required) | `"neon_yellow"` |
| `opponent.team_name` | Opponent team name | `"Away United"` |
| `opponent.outfield_color` | Opponent outfield kit color | `"red"` |
| `opponent.gk_color` | Opponent GK kit color (required for identification) | `"neon_green"` |

Valid color names: `white`, `black`, `grey`, `red`, `dark_red`, `orange`, `yellow`,
`neon_yellow`, `neon_green`, `neon_pink`, `green`, `dark_green`, `light_blue`,
`blue`, `dark_blue`, `navy`, `purple`, `pink`, `maroon`, `teal`, `cyan`,
`burgundy`, `gold`, `silver`, `brown`, `lime`, `coral`, `indigo`, `turquoise`.

## Check Status
```bash
# CLI
python infra/scripts/pipeline_cli.py status <job_id>
python infra/scripts/pipeline_cli.py status <job_id> --watch   # poll until done

# curl / UI
curl http://localhost:8080/jobs/{job_id}/status
open http://localhost:5555
```

## Expected Processing Times
- 90-min 4K (GPU): ~50 min detection + 5 min assembly
- 90-min 4K (CPU): ~3-4 hours

## Retrieve Output
```bash
curl http://localhost:8080/reels/{job_id}/keeper
curl -OJ http://localhost:8080/reels/{job_id}/keeper/download
```
Also available directly on NAS at NAS_OUTPUT_PATH/{job_id}/.
