from pathlib import Path
import csv
import random
from collections import defaultdict

from ultralytics import YOLO
import yaml
from PIL import Image, ImageDraw, ImageFont

# Read-only validation diagnostics. Does not train or modify model weights.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent.parent
MODEL_PATH = BACKEND_DIR / "models" / "best.pt"
DATA_YAML = PROJECT_DIR / "Dataset_V3_Detection3Class_CLEAN" / "data.yaml"
OUTPUT_DIR = BACKEND_DIR / "yolo_validation_error_report"

CONFIDENCE = 0.25
NMS_IOU = 0.45
IMAGE_SIZE = 640
MAX_IMAGES = 60  # Increase if desired; 60 gives a useful first audit.
SEED = 17
CLASS_NAMES = ["Apple", "Orange", "Pear"]

def resolve_dataset_path(yaml_path: Path, data: dict) -> Path:
    root_value = data.get("path")
    if root_value:
        root = Path(root_value)
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
    else:
        root = yaml_path.parent
    return root

def load_labels(label_path: Path, image_width: int, image_height: int):
    """Load YOLO normalized xywh labels and return pixel xyxy boxes + class IDs."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if cls not in (0, 1, 2):
            continue
        x1 = (xc - w / 2) * image_width
        y1 = (yc - h / 2) * image_height
        x2 = (xc + w / 2) * image_width
        y2 = (yc + h / 2) * image_height
        boxes.append((cls, (x1, y1, x2, y2)))
    return boxes

def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def get_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

def main():
    print("=" * 76)
    print("YOLO VALIDATION ERROR REPORT")
    print("=" * 76)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset YAML: {DATA_YAML}")
    print(f"Confidence={CONFIDENCE:.2f}, NMS IoU={NMS_IOU:.2f}")
    print(f"Maximum validation images to inspect: {MAX_IMAGES}")
    print("Read-only: model weights will not be changed.\n")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}\n"
            "Edit DATA_YAML near the top of this script to your actual data.yaml path."
        )

    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    root = resolve_dataset_path(DATA_YAML, data)
    val_entry = data.get("val", "valid/images")
    val_images_dir = Path(val_entry)
    if not val_images_dir.is_absolute():
        val_images_dir = root / val_images_dir
    val_images_dir = val_images_dir.resolve()

    if not val_images_dir.is_dir():
        raise FileNotFoundError(
            f"Validation image folder not found: {val_images_dir}\n"
            "Check the 'path' and 'val' entries in data.yaml."
        )

    image_files = sorted(
        p for p in val_images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if not image_files:
        raise RuntimeError(f"No validation images found in {val_images_dir}")

    # Deterministic sample, capped to keep the first diagnostic manageable.
    if len(image_files) > MAX_IMAGES:
        rng = random.Random(SEED)
        image_files = sorted(rng.sample(image_files, MAX_IMAGES))

    labels_dir = None
    val_str = str(val_entry).replace("\\", "/")
    if "/images" in val_str:
        labels_entry = val_str.replace("/images", "/labels", 1)
    elif val_str.endswith("images"):
        labels_entry = val_str[:-len("images")] + "labels"
    else:
        labels_entry = "valid/labels"
    labels_dir = Path(labels_entry)
    if not labels_dir.is_absolute():
        labels_dir = root / labels_dir
    labels_dir = labels_dir.resolve()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated_dir = OUTPUT_DIR / "annotated_examples"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    csv_rows = []
    totals = defaultdict(int)
    review_count = 0

    for index, image_path in enumerate(image_files, 1):
        try:
            with Image.open(image_path) as opened:
                original = opened.convert("RGB")
        except Exception as exc:
            print(f"Skipping unreadable image {image_path}: {exc}")
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        gt = load_labels(label_path, original.width, original.height)

        result = model.predict(
            source=original,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            imgsz=IMAGE_SIZE,
            verbose=False
        )[0]

        predictions = []
        if result.boxes is not None:
            for box in result.boxes:
                coords = tuple(float(x) for x in box.xyxy[0].cpu().tolist())
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                predictions.append({"cls": cls, "box": coords, "conf": conf})

        # Greedy same-class matching by IoU, highest confidence first.
        unmatched_gt = set(range(len(gt)))
        matched_predictions = set()
        for pred_idx in sorted(range(len(predictions)), key=lambda i: predictions[i]["conf"], reverse=True):
            pred = predictions[pred_idx]
            candidates = [
                (iou_xyxy(pred["box"], gt[gt_idx][1]), gt_idx)
                for gt_idx in unmatched_gt
                if gt[gt_idx][0] == pred["cls"]
            ]
            best_iou, best_gt_idx = max(candidates, default=(0.0, None))
            if best_gt_idx is not None and best_iou >= 0.5:
                unmatched_gt.remove(best_gt_idx)
                matched_predictions.add(pred_idx)

        false_positive_indices = set(range(len(predictions))) - matched_predictions
        false_negative_indices = unmatched_gt
        wrong_class_indices = set()
        # A prediction overlapping a ground-truth fruit but with the wrong class
        # is labelled as a likely class error for manual review.
        for pred_idx in false_positive_indices:
            pred = predictions[pred_idx]
            candidates = [
                (iou_xyxy(pred["box"], gt_i[1]), gt_i)
                for gt_i in range(len(gt))
            ]
            best_iou, best_gt_i = max(candidates, default=(0.0, None))
            if best_gt_i is not None and best_iou >= 0.5 and gt[best_gt_i][0] != pred["cls"]:
                wrong_class_indices.add(pred_idx)

        totals["images"] += 1
        totals["ground_truth_boxes"] += len(gt)
        totals["predicted_boxes"] += len(predictions)
        totals["matched_boxes"] += len(matched_predictions)
        totals["false_positives"] += len(false_positive_indices)
        totals["false_negatives"] += len(false_negative_indices)
        totals["likely_class_errors"] += len(wrong_class_indices)

        # Flag images with likely class errors, false positives, or missed GT.
        interesting = bool(false_positive_indices or false_negative_indices)
        if interesting:
            review_count += 1

        for pred_idx, pred in enumerate(predictions):
            if pred_idx in matched_predictions:
                status = "matched"
            elif pred_idx in wrong_class_indices:
                status = "likely_wrong_class"
            else:
                status = "false_positive_or_duplicate"
            csv_rows.append({
                "image": str(image_path),
                "label_file": str(label_path),
                "prediction_class": CLASS_NAMES[pred["cls"]],
                "prediction_confidence": round(pred["conf"], 4),
                "prediction_x1": round(pred["box"][0], 2),
                "prediction_y1": round(pred["box"][1], 2),
                "prediction_x2": round(pred["box"][2], 2),
                "prediction_y2": round(pred["box"][3], 2),
                "status": status,
                "ground_truth_boxes_in_image": len(gt),
            })

        # Save an annotated contact-sheet-style individual image for any errors.
        if interesting:
            canvas = original.copy()
            draw = ImageDraw.Draw(canvas)
            font = get_font(14)

            for cls, box in gt:
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle((x1, y1, x2, y2), outline=(0, 180, 255), width=3)
                draw.text((x1, max(0, y1 - 16)), f"GT {CLASS_NAMES[cls]}", fill=(0, 120, 200), font=font)

            for pred_idx, pred in enumerate(predictions):
                x1, y1, x2, y2 = map(int, pred["box"])
                if pred_idx in matched_predictions:
                    color = (30, 180, 70)
                    status = "OK"
                elif pred_idx in wrong_class_indices:
                    color = (255, 140, 0)
                    status = "CLASS?"
                else:
                    color = (230, 40, 40)
                    status = "FP?"
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                draw.text(
                    (x1, min(canvas.height - 16, y2 + 2)),
                    f"{status} {CLASS_NAMES[pred['cls']]} {pred['conf']:.2f}",
                    fill=color,
                    font=font,
                )

            out_path = annotated_dir / f"{index:03d}_{image_path.stem}.jpg"
            canvas.save(out_path, quality=90)

        if index % 10 == 0 or index == len(image_files):
            print(
                f"Processed {index}/{len(image_files)} | "
                f"GT={totals['ground_truth_boxes']} "
                f"Pred={totals['predicted_boxes']} "
                f"Matched={totals['matched_boxes']} "
                f"FP={totals['false_positives']} "
                f"FN={totals['false_negatives']}"
            )

    summary_path = OUTPUT_DIR / "prediction_audit.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else [
            "image", "label_file", "prediction_class", "prediction_confidence",
            "prediction_x1", "prediction_y1", "prediction_x2", "prediction_y2",
            "status", "ground_truth_boxes_in_image"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 76)
    print("AUDIT COMPLETE")
    print(f"Images inspected: {totals['images']}")
    print(f"Ground-truth boxes: {totals['ground_truth_boxes']}")
    print(f"Predicted boxes: {totals['predicted_boxes']}")
    print(f"Matched predictions (same class, IoU >= 0.50): {totals['matched_boxes']}")
    print(f"Unmatched predictions / possible false positives: {totals['false_positives']}")
    print(f"Likely wrong-class overlaps: {totals['likely_class_errors']}")
    print(f"Missed ground-truth boxes: {totals['false_negatives']}")
    print(f"Images flagged for review: {review_count}")
    print(f"Prediction CSV: {summary_path}")
    print(f"Annotated examples: {annotated_dir}")
    print("\nBlue boxes = ground truth; green = matched; orange = likely class mismatch; red = unmatched prediction.")
    print("These are heuristic audit labels, not a substitute for full mAP evaluation.")
    print("=" * 76)

if __name__ == "__main__":
    main()
