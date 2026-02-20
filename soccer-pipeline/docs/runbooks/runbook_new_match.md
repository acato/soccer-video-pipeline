# Runbook: Processing a New Match

## Automatic (Watcher) Flow
1. Drop MP4 to NAS_MOUNT_PATH  
2. Watcher detects file after 30s stable size → job auto-created
3. Monitor at http://localhost:5555

## Manual Submission
```bash
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{"nas_path": "matches/game.mp4", "reel_types": ["goalkeeper", "highlights"]}'
```

## Check Status
```bash
curl http://localhost:8080/jobs/{job_id}/status
open http://localhost:5555
```

## Expected Processing Times
- 90-min 4K (GPU): ~50 min detection + 5 min assembly
- 90-min 4K (CPU): ~3-4 hours

## Retrieve Output
```bash
curl http://localhost:8080/reels/{job_id}/goalkeeper
curl -OJ http://localhost:8080/reels/{job_id}/goalkeeper/download
```
Also available directly on NAS at NAS_OUTPUT_PATH/{job_id}/.
