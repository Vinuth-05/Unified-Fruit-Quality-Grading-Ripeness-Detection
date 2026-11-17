from pathlib import Path
import csv
from ultralytics import YOLO

# Run this script from Review01_App/backend.
# It evaluates the existing YOLO checkpoint on the VALIDATION split only.
# It does not train, modify, or overwrite any model weights.

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "models" / "best.pt"
DATA_YAML = Path(
    r"C:\Users\bdhul\Documents\Fruit_Grading_Project"
    r"\Dataset_V3_Detection3Class_CLEAN\data.yaml"
)
OUTPUT_DIR = BACKEND_DIR / "yolo_threshold_evaluation"
CONFIDENCE_LEVELS = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640

def main():
    print("=" * 72)
    print("YOLO VALIDATION THRESHOLD EVALUATION")
    print("=" * 72)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nThis script evaluates the validation split only.")
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

    for conf in CONFIDENCE_LEVELS:
        print("\n" + "-" * 72)
        print(f"Evaluating confidence={conf:.2f}, NMS IoU={IOU_THRESHOLD:.2f}")
        metrics = model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=IMAGE_SIZE,
            conf=conf,
            iou=IOU_THRESHOLD,
            plots=False,
            verbose=False,
            project=str(OUTPUT_DIR),
            name=f"conf_{conf:.2f}",
            exist_ok=True,
        )

        box = metrics.box
        row = {
            "confidence_threshold": conf,
            "nms_iou_threshold": IOU_THRESHOLD,
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

    csv_path = OUTPUT_DIR / "threshold_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETE")
    print(f"Summary CSV: {csv_path}")
    print("Compare precision and recall together; a higher confidence")
    print("threshold may reduce false positives but also miss real fruits.")
    print("Do not select a threshold from one image alone.")
    print("=" * 72)

if __name__ == "__main__":
    main()
