"""
Evaluate YOLO confidence thresholds across the FULL validation split.

Run from the backend folder:
  python evaluate_yolo_confidence_full_validation.py

Uses existing best.pt and data.yaml. Does not train or modify model weights.
Compares confidence thresholds 0.25, 0.35, 0.40 with NMS IoU fixed at 0.45.
Writes summary and per-class metrics under:
  backend/yolo_confidence_full_validation/
"""

from pathlib import Path
import csv
import sys

import yaml
from ultralytics import YOLO


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent.parent
MODEL_PATH = BACKEND_DIR / "models" / "best.pt"
DATA_YAML = PROJECT_DIR / "Dataset_V3_Detection3Class_CLEAN" / "data.yaml"
OUTPUT_DIR = BACKEND_DIR / "yolo_confidence_full_validation"
THRESHOLDS = (0.25, 0.35, 0.40)
NMS_IOU = 0.45
IMAGE_SIZE = 640


def main():
    print("=" * 78)
    print("YOLO FULL VALIDATION CONFIDENCE THRESHOLD EVALUATION")
    print("=" * 78)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Confidence thresholds: {list(THRESHOLDS)}")
    print(f"Fixed NMS IoU: {NMS_IOU}")
    print("Full validation split only.")
    print("Read-only: this script does NOT train or modify best.pt.")
    print("=" * 78)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("val"):
        raise ValueError(f"Dataset YAML has no valid 'val' entry: {DATA_YAML}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    rows = []

    for conf in THRESHOLDS:
        print("\n" + "-" * 78)
        print(f"Evaluating confidence={conf:.2f}, NMS IoU={NMS_IOU:.2f}")
        metrics = model.val(
            data=str(DATA_YAML),
            split="val",
            conf=conf,
            iou=NMS_IOU,
            imgsz=IMAGE_SIZE,
            plots=True,
            save_json=False,
            verbose=True,
            project=str(OUTPUT_DIR / "ultralytics_runs"),
            name=f"conf_{conf:.2f}",
            exist_ok=True,
        )

        box = metrics.box
        precision = float(box.mp)
        recall = float(box.mr)
        map50 = float(box.map50)
        map5095 = float(box.map)

        rows.append({
            "confidence_threshold": conf,
            "nms_iou_threshold": NMS_IOU,
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50_95": map5095,
        })

        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"mAP50:     {map50:.4f}")
        print(f"mAP50-95:  {map5095:.4f}")

        names = model.names
        per_class_rows = []
        p = getattr(box, "p", None)
        r = getattr(box, "r", None)
        f1 = getattr(box, "f1", None)
        ap50 = getattr(box, "ap50", None)
        ap = getattr(box, "ap", None)
        class_indices = list(getattr(box, "ap_class_index", []))

        for i, class_id in enumerate(class_indices):
            class_name = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
            per_class_rows.append({
                "confidence_threshold": conf,
                "class_id": int(class_id),
                "class_name": class_name,
                "precision": float(p[i]) if p is not None and len(p) > i else "",
                "recall": float(r[i]) if r is not None and len(r) > i else "",
                "F1": float(f1[i]) if f1 is not None and len(f1) > i else "",
                "AP50": float(ap50[i]) if ap50 is not None and len(ap50) > i else "",
                "AP50_95": float(ap[i].mean()) if ap is not None and len(ap) > i else "",
            })

        per_class_path = OUTPUT_DIR / f"per_class_conf_{conf:.2f}.csv"
        with per_class_path.open("w", newline="", encoding="utf-8-sig") as f:
            fieldnames = [
                "confidence_threshold", "class_id", "class_name",
                "precision", "recall", "F1", "AP50", "AP50_95",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_class_rows)

        print(f"Per-class CSV: {per_class_path}")

    summary_path = OUTPUT_DIR / "confidence_threshold_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "confidence_threshold", "nms_iou_threshold",
            "precision", "recall", "mAP50", "mAP50_95",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("FULL VALIDATION EVALUATION COMPLETE")
    print(f"Summary CSV: {summary_path}")
    print(f"Per-class CSV files: {OUTPUT_DIR}")
    print("Ultralytics validation runs are saved under:")
    print(OUTPUT_DIR / "ultralytics_runs")
    print("Use full-split results to decide whether any threshold change is justified.")
    print("No model weights were changed.")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
