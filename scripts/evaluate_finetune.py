#!/usr/bin/env python3
"""
Evaluate a fine-tuned Qwen3-VL model against holdout test data.

Loads holdout JSONL, runs inference with the fine-tuned model (via vLLM API),
and computes per-class precision/recall/F1 + confusion matrix.

Usage:
    python scripts/evaluate_finetune.py \
        --holdout /data/soccer-finetune/stage2_holdout.jsonl \
        --vllm-url http://10.10.2.222:8000 \
        --model soccer-classifier \
        --output /data/soccer-finetune/eval_results.json

    # Compare base model vs fine-tuned model
    python scripts/evaluate_finetune.py \
        --holdout /data/soccer-finetune/stage2_holdout.jsonl \
        --vllm-url http://10.10.2.222:8000 \
        --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
        --output /data/soccer-finetune/eval_baseline.json
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


def load_holdout(path: Path) -> list[dict]:
    """Load holdout JSONL records."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_gt_label(record: dict) -> str:
    """Extract ground truth label from a training record."""
    meta = record.get("_meta", {})
    return meta.get("label", "none")


def parse_prediction(response_text: str) -> tuple[str, float]:
    """Parse model response into (predicted_label, confidence)."""
    # Try to parse as JSON
    text = response_text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    try:
        data = json.loads(text)
        return data.get("event", "none"), data.get("confidence", 0.5)
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON in the response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            return data.get("event", "none"), data.get("confidence", 0.5)
        except json.JSONDecodeError:
            pass

    return "parse_error", 0.0


def call_vllm(
    url: str,
    model: str,
    query: str,
    image_paths: list[str],
    base_dir: Path,
    timeout: int = 120,
) -> Optional[str]:
    """Call vLLM API with images."""
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx not installed. pip install httpx")
        sys.exit(1)

    content = []

    for img_path in image_paths:
        full_path = base_dir / img_path
        if not full_path.exists():
            continue
        with open(full_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    content.append({"type": "text", "text": query})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,
        "temperature": 0,
    }

    try:
        r = httpx.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        else:
            print(f"  vLLM error: {r.status_code} {r.text[:200]}")
            return None
    except Exception as e:
        print(f"  vLLM exception: {e}")
        return None


def compute_metrics(
    gt_labels: list[str],
    pred_labels: list[str],
    all_classes: list[str],
) -> dict:
    """Compute per-class precision, recall, F1 and confusion matrix."""
    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for gt, pred in zip(gt_labels, pred_labels):
        confusion[gt][pred] += 1

    # Per-class metrics
    per_class = {}
    for cls in all_classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in all_classes if other != cls)
        fn = sum(confusion[cls][other] for other in all_classes if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # Overall metrics
    total_correct = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
    overall_accuracy = total_correct / len(gt_labels) if gt_labels else 0.0

    # Macro average (unweighted mean across classes)
    macro_precision = sum(m["precision"] for m in per_class.values()) / len(per_class) if per_class else 0
    macro_recall = sum(m["recall"] for m in per_class.values()) / len(per_class) if per_class else 0
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(per_class) if per_class else 0

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": len(gt_labels),
        "total_correct": total_correct,
        "per_class": per_class,
        "confusion_matrix": {gt: dict(preds) for gt, preds in confusion.items()},
    }


def print_results(metrics: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nOverall accuracy: {metrics['overall_accuracy']:.1%} "
          f"({metrics['total_correct']}/{metrics['total_samples']})")
    print(f"Macro precision:  {metrics['macro_precision']:.1%}")
    print(f"Macro recall:     {metrics['macro_recall']:.1%}")
    print(f"Macro F1:         {metrics['macro_f1']:.1%}")

    print(f"\n{'Class':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'Support':>8}")
    print("-" * 70)

    for cls, m in sorted(metrics["per_class"].items(), key=lambda x: -x[1]["support"]):
        if m["support"] == 0 and m["fp"] == 0:
            continue
        print(f"{cls:<18} {m['precision']:>6.1%} {m['recall']:>6.1%} {m['f1']:>6.1%} "
              f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['support']:>8}")

    # Goal-specific analysis (most important)
    if "goal" in metrics["per_class"]:
        g = metrics["per_class"]["goal"]
        print(f"\n--- GOAL DETECTION (critical metric) ---")
        print(f"  Precision: {g['precision']:.1%} ({g['tp']} TP, {g['fp']} FP)")
        print(f"  Recall:    {g['recall']:.1%} ({g['tp']} TP, {g['fn']} FN)")
        print(f"  F1:        {g['f1']:.1%}")

    # Confusion matrix for key classes
    conf = metrics.get("confusion_matrix", {})
    key_classes = ["goal", "save_catch", "save_parry", "shot_on_target", "shot_off_target"]
    present = [c for c in key_classes if c in conf]

    if present:
        print(f"\n--- Confusion Matrix (key classes) ---")
        header = f"{'GT \\ Pred':<18}" + "".join(f"{c[:12]:>13}" for c in present)
        print(header)
        print("-" * (18 + 13 * len(present)))
        for gt_cls in present:
            row = f"{gt_cls:<18}"
            for pred_cls in present:
                count = conf.get(gt_cls, {}).get(pred_cls, 0)
                row += f"{count:>13}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned VLM")
    parser.add_argument("--holdout", required=True, help="Holdout JSONL file")
    parser.add_argument("--vllm-url", default="http://10.10.2.222:8000",
                       help="vLLM API URL")
    parser.add_argument("--model", default="soccer-classifier",
                       help="Model name (use LoRA module name for fine-tuned)")
    parser.add_argument("--base-dir", default=None,
                       help="Base directory for resolving image paths (default: parent of holdout file)")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of samples to evaluate (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Parse GT labels without running inference")

    args = parser.parse_args()

    holdout_path = Path(args.holdout)
    base_dir = Path(args.base_dir) if args.base_dir else holdout_path.parent

    records = load_holdout(holdout_path)
    if args.limit > 0:
        records = records[:args.limit]

    print(f"Loaded {len(records)} holdout samples")
    print(f"Model: {args.model}")
    print(f"vLLM URL: {args.vllm_url}")
    print(f"Base dir: {base_dir}")

    # Extract GT labels
    gt_labels = [extract_gt_label(r) for r in records]
    all_classes = sorted(set(gt_labels))

    print(f"\nGT class distribution:")
    for cls, count in Counter(gt_labels).most_common():
        print(f"  {cls}: {count}")

    if args.dry_run:
        print("\n[Dry run -- skipping inference]")
        return

    # Run inference
    pred_labels = []
    pred_confidences = []
    errors = 0

    start_time = time.time()

    for i, record in enumerate(records):
        gt = gt_labels[i]
        query = record["query"]
        images = record.get("images", [])

        print(f"\r  Evaluating {i+1}/{len(records)} (GT: {gt})...", end="", flush=True)

        response = call_vllm(
            args.vllm_url, args.model, query, images, base_dir
        )

        if response is None:
            pred_labels.append("error")
            pred_confidences.append(0.0)
            errors += 1
            continue

        pred_label, confidence = parse_prediction(response)
        pred_labels.append(pred_label)
        pred_confidences.append(confidence)

    elapsed = time.time() - start_time
    print(f"\n\nInference complete: {len(records)} samples in {elapsed:.1f}s "
          f"({elapsed/len(records):.1f}s/sample)")

    if errors > 0:
        print(f"  {errors} inference errors")

    # Add error and parse_error to all_classes if present
    all_classes = sorted(set(gt_labels + pred_labels))

    # Compute metrics
    metrics = compute_metrics(gt_labels, pred_labels, all_classes)
    metrics["model"] = args.model
    metrics["holdout_file"] = str(holdout_path)
    metrics["inference_time_sec"] = round(elapsed, 1)
    metrics["errors"] = errors

    print_results(metrics)

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
