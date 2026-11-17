"""
Compare YOLO detection confidence thresholds on images flagged by the validation audit.

Run from the backend folder:
    python compare_yolo_flagged_thresholds.py

This script:
- Loads the existing best.pt (read-only; does not train or modify weights).
- Reads yolo_validation_error_report/prediction_audit.csv.
- Selects images with audit statuses other than "matched".
- Runs each selected image at confidence thresholds 0.25, 0.35, and 0.40.
- Saves per-threshold annotated images, a combined side-by-side comparison, and a CSV.
"""

from pathlib import Path
import csv
import sys

import cv2
import yaml
from ultralytics import YOLO


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent.parent
MODEL_PATH = BACKEND_DIR / "models" / "best.pt"
DATA_YAML = PROJECT_DIR / "Dataset_V3_Detection3Class_CLEAN" / "data.yaml"
AUDIT_CSV = BACKEND_DIR / "yolo_validation_error_report" / "prediction_audit.csv"
OUTPUT_DIR = BACKEND_DIR / "yolo_threshold_comparison"
THRESHOLDS = (0.25, 0.35, 0.40)
NMS_IOU = 0.45
IMAGE_SIZE = 640
MAX_IMAGES = 60


def resolve_image_path(raw_path: str, data_yaml_path: Path) -> Path:
    """Resolve the path recorded in the audit CSV, with a dataset fallback."""
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    # The CSV normally contains an absolute path from the user's Windows machine.
    # Fall back to the current dataset's validation image folder by basename.
    dataset_root = data_yaml_path.parent
    for split_name in ("valid", "val"):
        candidate = dataset_root / split_name / "images" / Path(raw_path).name
        if candidate.exists():
            return candidate

    # Last fallback: use the configured path and validation folder from data.yaml.
    try:
        config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
        configured_root = Path(str(config.get("path", dataset_root)))
        if not configured_root.is_absolute():
            configured_root = (data_yaml_path.parent / configured_root).resolve()
        val_entry = str(config.get("val", "valid/images"))
        val_dir = Path(val_entry)
        if not val_dir.is_absolute():
            val_dir = configured_root / val_dir
        candidate = val_dir / Path(raw_path).name
        if candidate.exists():
            return candidate
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not locate image '{Path(raw_path).name}'. "
        f"Checked the recorded path and validation folders under {dataset_root}"
    )


def read_flagged_images(csv_path: Path, yaml_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Audit CSV not found: {csv_path}\n"
            "Run audit_yolo_validation_errors_fixed.py first."
        )

    grouped = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"image", "status"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Audit CSV must contain columns {sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            status = (row.get("status") or "").strip()
            if status.lower() == "matched":
                continue
            raw_image = (row.get("image") or "").strip()
            if not raw_image:
                continue
            key = Path(raw_image).name
            grouped.setdefault(key, {"raw_path": raw_image, "statuses": set()})
            grouped[key]["statuses"].add(status or "unspecified")

    selected = []
    for basename, info in grouped.items():
        try:
            resolved = resolve_image_path(info["raw_path"], yaml_path)
            selected.append((basename, resolved, sorted(info["statuses"])))
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")

    return selected[:MAX_IMAGES]


def main():
    print("=" * 76)
    print("YOLO FLAGGED-IMAGE CONFIDENCE THRESHOLD COMPARISON")
    print("=" * 76)
    print(f"Model: {MODEL_PATH}")
    print(f"Audit CSV: {AUDIT_CSV}")
    print(f"Confidence thresholds: {list(THRESHOLDS)}")
    print(f"NMS IoU: {NMS_IOU}")
    print("Read-only: model weights will not be changed.")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL_PATH}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    flagged = read_flagged_images(AUDIT_CSV, DATA_YAML)
    if not flagged:
        print("No flagged images were found or resolved from the audit CSV.")
        return

    print(f"Flagged images resolved: {len(flagged)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated_dir = OUTPUT_DIR / "annotated"
    comparison_dir = OUTPUT_DIR / "side_by_side"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    names = model.names
    records = []

    for image_index, (basename, image_path, statuses) in enumerate(flagged, start=1):
        print(f"[{image_index}/{len(flagged)}] {basename} | audit={','.join(statuses)}")
        panels = []
        source = cv2.imread(str(image_path))
        if source is None:
            print(f"  WARNING: OpenCV could not read {image_path}")
            continue

        for conf in THRESHOLDS:
            results = model.predict(
                source=str(image_path),
                conf=conf,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                verbose=False,
                save=False,
            )
            result = results[0]
            plotted = result.plot()
            panel_path = annotated_dir / f"{image_path.stem}_conf_{conf:.2f}.jpg"
            cv2.imwrite(str(panel_path), plotted)
            panels.append((conf, plotted))

            boxes = result.boxes
            detections = 0 if boxes is None else len(boxes)
            class_counts = {}
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    xyxy = box.xyxy[0].tolist()
                    records.append({
                        "image": str(image_path),
                        "image_name": basename,
                        "audit_status": ";".join(statuses),
                        "confidence_threshold": conf,
                        "nms_iou": NMS_IOU,
                        "prediction_class": class_name,
                        "prediction_confidence": round(confidence, 4),
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2),
                        "detections_in_image": detections,
                    })

            print(f"  conf={conf:.2f}: {detections} detection(s) | {class_counts}")

        # Resize panels to equal height, label each panel, and join horizontally.
        labeled_panels = []
        target_height = 480
        for conf, panel in panels:
            h, w = panel.shape[:2]
            resized = cv2.resize(panel, (max(1, int(w * target_height / h)), target_height))
            header = resized.copy()
            cv2.rectangle(header, (0, 0), (header.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(
                header, f"Confidence {conf:.2f} | NMS IoU {NMS_IOU:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            labeled_panels.append(header)

        combined = cv2.hconcat(labeled_panels)
        combined_path = comparison_dir / f"{image_path.stem}_comparison.jpg"
        cv2.imwrite(str(combined_path), combined)

    report_path = OUTPUT_DIR / "threshold_comparison.csv"
    fieldnames = [
        "image", "image_name", "audit_status", "confidence_threshold", "nms_iou",
        "prediction_class", "prediction_confidence", "x1", "y1", "x2", "y2",
        "detections_in_image",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print()
    print("=" * 76)
    print("COMPARISON COMPLETE")
    print(f"Images compared: {len(flagged)}")
    print(f"Prediction CSV: {report_path}")
    print(f"Annotated panels: {annotated_dir}")
    print(f"Side-by-side comparisons: {comparison_dir}")
    print("Compare whether unwanted boxes disappear without losing valid fruit detections.")
    print("This diagnostic does not choose a threshold automatically or modify best.pt.")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
