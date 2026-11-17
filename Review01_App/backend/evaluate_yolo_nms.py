from pathlib import Path
import csv
from ultralytics import YOLO

# Validation-only NMS IoU comparison. Does not train or modify model weights.
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "models" / "best.pt"
DATA_YAML = Path(
    r"C:\Users\bdhul\Documents\Fruit_Grading_Project"
    r"\Dataset_V3_Detection3Class_CLEAN\data.yaml"
)
OUTPUT_DIR = BACKEND_DIR / "yolo_nms_evaluation"
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_LEVELS = [0.35, 0.45, 0.55]
IMAGE_SIZE = 640

def main():
    print("=" * 72)
    print("YOLO NMS IoU VALIDATION EVALUATION")
    print("=" * 72)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Fixed confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print("NMS IoU thresholds:", NMS_IOU_LEVELS)
    print("\nThis evaluates the validation split only.")
    print("It does NOT train or change best.pt.\n")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL_PATH}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            "Dataset YAML not found:\n"
            f"{DATA_YAML}\n\n"
            "Edit DATA_YAML at the top of this script to point to your "
            "Dataset_V3_Detection3Class_CLEAN\\data.yaml file."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    rows = []

    for nms_iou in NMS_IOU_LEVELS:
        print("\n" + "-" * 72)
        print(
            f"Evaluating confidence={CONFIDENCE_THRESHOLD:.2f}, "
            f"NMS IoU={nms_iou:.2f}"
        )
        metrics = model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=nms_iou,
            plots=False,
            verbose=False,
            project=str(OUTPUT_DIR),
            name=f"nms_iou_{nms_iou:.2f}",
            exist_ok=True,
        )

        box = metrics.box
        row = {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": nms_iou,
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
        }
        rows.append(row)

        print(f"Precision: {row['precision']:.4f}")
        print(f"Recall:    {row['recall']:.4f}")
        print(f"mAP50:     {row['mAP50']:.4f}")
        print(f"mAP50-95:  {row['mAP50_95']:.4f}")

    csv_path = OUTPUT_DIR / "nms_iou_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETE")
    print(f"Summary CSV: {csv_path}")
    print(
        "Interpretation: lower NMS IoU suppresses overlapping boxes more "
        "aggressively; higher NMS IoU allows more overlapping detections. "
        "Compare metrics and inspect examples before selecting a value."
    )
    print("=" * 72)

if __name__ == "__main__":
    main()
