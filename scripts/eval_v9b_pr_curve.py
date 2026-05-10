"""Per-frame precision/recall curve for v9b across conf thresholds.

For each val frame:
  - run YOLO at conf=0.001 (catch all candidates)
  - compare each detection to GT (single ball center, 0.012 normalized box)
  - a det matches GT if center distance < 0.04 (4% of frame width — ~77 px)

Outputs:
  - PR table by conf threshold
  - sample annotated frames at conf=0.05 to /tmp/v9b_qual/
"""
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

VAL_DIR = Path("/mnt/transit/soccer-finetune/yolo_ball_v9/images/val")
LBL_DIR = Path("/mnt/transit/soccer-finetune/yolo_ball_v9/labels/val")
WEIGHTS = "/mnt/transit/soccer-finetune/yolo_ball_v9/weights/v9b_best.pt"
QUAL_OUT = Path("/mnt/transit/soccer-finetune/yolo_ball_v9/qual_v9b")
MATCH_DIST = 0.04  # normalized

if QUAL_OUT.exists():
    shutil.rmtree(QUAL_OUT)
QUAL_OUT.mkdir(parents=True)

model = YOLO(WEIGHTS)

# All candidates per frame
all_dets = []  # list of (frame, conf, x_n, y_n, has_gt, gt_center)
for jpg in sorted(VAL_DIR.glob("*.jpg")):
    lbl = LBL_DIR / (jpg.stem + ".txt")
    has_gt = False
    gt_x = gt_y = None
    if lbl.exists() and lbl.read_text().strip():
        parts = lbl.read_text().strip().split()
        gt_x, gt_y = float(parts[1]), float(parts[2])
        has_gt = True
    res = model.predict(str(jpg), imgsz=1920, conf=0.001, verbose=False)[0]
    H, W = res.orig_shape
    if len(res.boxes) == 0:
        all_dets.append((jpg.stem, None, None, None, has_gt, (gt_x, gt_y)))
        continue
    for b in res.boxes:
        conf = float(b.conf[0])
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        all_dets.append((jpg.stem, conf, cx, cy, has_gt, (gt_x, gt_y)))

# PR curve
total_pos = sum(1 for _, _, _, _, has_gt, _ in all_dets if has_gt and (None, None) == (None, None))  # gt frames
n_pos_frames = len({f for f, _, _, _, has_gt, _ in all_dets if has_gt})
print(f"frames with gt ball: {n_pos_frames}")
print(f"frames total: {len({f for f, _, _, _, _, _ in all_dets})}")
print()
print(f"{'conf':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'frames_w_det':>14}")

frames_with_gt = {f for f, _, _, _, has_gt, _ in all_dets if has_gt}
gt_centers = {f: (gx, gy) for f, _, _, _, has_gt, (gx, gy) in all_dets if has_gt}

for conf_t in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    tp_frames = set()  # frames where at least one det matched gt
    fp_count = 0
    frames_with_det = set()
    for f, conf, cx, cy, has_gt, (gx, gy) in all_dets:
        if conf is None or conf < conf_t:
            continue
        frames_with_det.add(f)
        if has_gt and ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5 < MATCH_DIST:
            tp_frames.add(f)
        else:
            fp_count += 1
    tp = len(tp_frames)
    fn = len(frames_with_gt) - tp
    fp = fp_count
    P = tp / max(1, tp + fp)
    R = tp / max(1, tp + fn)
    print(f"{conf_t:>6.2f} {tp:>4} {fp:>4} {fn:>4} {P:>6.3f} {R:>6.3f} {len(frames_with_det):>14}")

# Save annotated samples at conf=0.05
import cv2
ann_count = 0
for jpg in sorted(VAL_DIR.glob("*.jpg")):
    if ann_count >= 24:
        break
    res = model.predict(str(jpg), imgsz=1920, conf=0.05, verbose=False)[0]
    img = cv2.imread(str(jpg))
    H, W = img.shape[:2]
    lbl = LBL_DIR / (jpg.stem + ".txt")
    has_gt = lbl.exists() and lbl.read_text().strip()
    if has_gt:
        parts = lbl.read_text().strip().split()
        gx, gy = float(parts[1]), float(parts[2])
        gxp, gyp = int(gx * W), int(gy * H)
        cv2.circle(img, (gxp, gyp), 25, (0, 255, 0), 2)
        cv2.putText(img, "GT", (gxp + 28, gyp + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        c = float(b.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{c:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(QUAL_OUT / jpg.name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    ann_count += 1
print(f"\n{ann_count} qualitative samples at {QUAL_OUT}")
