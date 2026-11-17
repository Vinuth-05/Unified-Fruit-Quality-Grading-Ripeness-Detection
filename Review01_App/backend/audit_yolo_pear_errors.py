"""
Pear-specific YOLO validation error audit.
- Does NOT train or modify model weights.
- Audits missed pears, pear class confusions, and false-positive pear detections.
- Writes CSV reports and annotated examples for manual review.

Run from your backend folder:
python audit_yolo_pear_errors.py --weights models/best.pt --data "C:/Users/bdhul/Documents/Fruit_Grading_Project/Dataset_V3_Detection3Class_CLEAN/data.yaml"
"""

import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


CLASS_NAMES = {0: "Apple", 1: "Orange", 2: "Pear"}
PEAR_ID = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Audit pear-specific YOLO validation errors.")
    parser.add_argument("--weights", default="models/best.pt", help="Path to YOLO best.pt")
    parser.add_argument("--data", default=r"C:\Users\bdhul\Documents\Fruit_Grading_Project\Dataset_V3_Detection3Class_CLEAN\data.yaml",
                        help="Path to dataset data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "valid", "test"],
                        help="Dataset split to audit (default: val)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50,
                        help="IoU threshold for considering boxes overlapping/matched")
    parser.add_argument("--max-examples", type=int, default=60,
                        help="Maximum annotated images to save; 0 saves all flagged images")
    parser.add_argument("--out", default="pear_error_audit",
                        help="Output folder")
    return parser.parse_args()


def resolve_dataset_root(data_yaml_path, config):
    configured = config.get("path")
    if configured:
        root = Path(str(configured).replace("/", os.sep))
        if root.exists():
            return root.resolve()

        # If YAML came from another operating system, try paths relative to YAML.
        yaml_parent = data_yaml_path.parent
        candidate = (yaml_parent / root).resolve()
        if candidate.exists():
            return candidate

        # Common case: data.yaml is inside the dataset root.
        if (yaml_parent / "train").exists() or (yaml_parent / "valid").exists() or (yaml_parent / "val").exists():
            return yaml_parent.resolve()

    yaml_parent = data_yaml_path.parent
    if (yaml_parent / "train").exists() or (yaml_parent / "valid").exists() or (yaml_parent / "val").exists():
        return yaml_parent.resolve()

    raise FileNotFoundError(
        "Could not resolve dataset root from data.yaml. Check the 'path' value or place data.yaml inside the dataset folder."
    )


def split_value(config, requested):
    candidates = [requested]
    if requested == "val":
        candidates += ["valid"]
    elif requested == "valid":
        candidates += ["val"]
    for key in candidates:
        value = config.get(key)
        if value:
            return str(value)
    raise KeyError(f"No '{requested}' or compatible split entry found in data.yaml.")


def image_dir_from_config(root, split_entry):
    p = Path(split_entry)
    if p.is_absolute() and p.exists():
        return p.resolve()
    candidate = (root / p).resolve()
    if candidate.exists():
        return candidate
    # Accommodate a split entry that includes a duplicated dataset folder name.
    for part in p.parts:
        if part.lower() in ("train", "valid", "val", "test"):
            candidate = root.joinpath(*p.parts[p.parts.index(part):]).resolve()
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Image folder does not exist: {candidate}")


def label_dir_from_images(image_dir):
    # Standard YOLO layout: .../valid/images -> .../valid/labels
    if image_dir.name.lower() == "images":
        return image_dir.parent / "labels"
    return image_dir.parent / "labels"


def read_yolo_labels(label_path, width, height):
    boxes = []
    if not label_path.exists():
        return boxes
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width - 1), x2))
        y2 = max(0.0, min(float(height - 1), y2))
        boxes.append({"cls": cls, "xyxy": [x1, y1, x2, y2], "line": line_no})
    return boxes


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fmt_box(box):
    return ",".join(str(round(float(v), 1)) for v in box)


def draw_box(image, box, color, label, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    text_y = max(18, y1 - 6)
    cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    data_yaml_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    examples_dir = out_dir / "annotated_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

    config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    root = resolve_dataset_root(data_yaml_path, config)
    split_entry = split_value(config, args.split)
    image_dir = image_dir_from_config(root, split_entry)
    labels_dir = label_dir_from_images(image_dir)

    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found under {image_dir}")

    print(f"Dataset root : {root}")
    print(f"Split images : {image_dir}")
    print(f"Split labels : {labels_dir}")
    print(f"Images       : {len(image_paths)}")
    print(f"Weights      : {weights_path}")
    print(f"Settings     : conf={args.conf}, NMS IoU={args.nms_iou}, match IoU={args.match_iou}")

    model = YOLO(str(weights_path))
    rows = []
    flagged_images = []
    totals = {
        "images_checked": 0,
        "gt_pear_boxes": 0,
        "missed_pear_boxes": 0,
        "pear_class_confusions": 0,
        "false_positive_pear_boxes": 0,
        "matched_pear_boxes": 0,
    }

    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue
        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_boxes = read_yolo_labels(label_path, width, height)

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            verbose=False,
            save=False,
        )[0]

        pred_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for b, c, conf in zip(xyxy, clss, confs):
                pred_boxes.append({"cls": int(c), "conf": float(conf), "xyxy": [float(v) for v in b]})

        totals["images_checked"] += 1
        gt_pear = [g for g in gt_boxes if g["cls"] == PEAR_ID]
        pred_pear = [p for p in pred_boxes if p["cls"] == PEAR_ID]
        totals["gt_pear_boxes"] += len(gt_pear)

        image_rows = []
        matched_gt_pear = set()
        matched_pred_pear = set()

        # Class-agnostic overlap scan: catches a non-pear prediction on a pear GT.
        for gi, gt in enumerate(gt_pear):
            overlaps = [(box_iou(gt["xyxy"], p["xyxy"]), pi, p)
                        for pi, p in enumerate(pred_boxes)]
            overlaps = [item for item in overlaps if item[0] >= args.match_iou]
            pear_overlaps = [item for item in overlaps if item[2]["cls"] == PEAR_ID]
            wrong_overlaps = [item for item in overlaps if item[2]["cls"] != PEAR_ID]

            if pear_overlaps:
                best_iou, pi, pred = max(pear_overlaps, key=lambda x: x[0])
                matched_gt_pear.add(gi)
                matched_pred_pear.add(pi)
                totals["matched_pear_boxes"] += 1
            elif wrong_overlaps:
                best_iou, pi, pred = max(wrong_overlaps, key=lambda x: x[0])
                totals["pear_class_confusions"] += 1
                image_rows.append({
                    "image": str(image_path), "issue_type": "PEAR_GT_WRONG_CLASS_PREDICTION",
                    "gt_class": "Pear", "pred_class": CLASS_NAMES.get(pred["cls"], str(pred["cls"])),
                    "confidence": round(pred["conf"], 4), "iou": round(best_iou, 4),
                    "gt_box_xyxy": fmt_box(gt["xyxy"]), "pred_box_xyxy": fmt_box(pred["xyxy"]),
                    "details": "Ground-truth pear overlaps a prediction of another class; inspect for class confusion."
                })
            else:
                totals["missed_pear_boxes"] += 1
                image_rows.append({
                    "image": str(image_path), "issue_type": "MISSED_PEAR",
                    "gt_class": "Pear", "pred_class": "None",
                    "confidence": "", "iou": round(max((box_iou(gt["xyxy"], p["xyxy"]) for p in pred_boxes), default=0.0), 4),
                    "gt_box_xyxy": fmt_box(gt["xyxy"]), "pred_box_xyxy": "",
                    "details": "No predicted box of any class reached the configured IoU threshold."
                })

        # False-positive pear predictions: no pear GT overlaps at the threshold.
        for pi, pred in enumerate(pred_pear):
            # pred_pear index differs from pred_boxes index; evaluate directly.
            best_pear_iou = max((box_iou(pred["xyxy"], gt["xyxy"]) for gt in gt_pear), default=0.0)
            if best_pear_iou >= args.match_iou:
                continue
            best_any = max(
                ((box_iou(pred["xyxy"], gt["xyxy"]), gt) for gt in gt_boxes),
                key=lambda x: x[0],
                default=(0.0, None),
            )
            best_iou, best_gt = best_any
            if best_gt is None:
                issue = "FALSE_POSITIVE_PEAR_NO_GT"
                gt_name = "None"
                detail = "Pear predicted in an image with no annotated ground-truth box."
            elif best_gt["cls"] != PEAR_ID and best_iou >= args.match_iou:
                issue = "WRONG_CLASS_PEAR_ON_OTHER_FRUIT"
                gt_name = CLASS_NAMES.get(best_gt["cls"], str(best_gt["cls"]))
                detail = "Pear prediction overlaps a different ground-truth fruit class."
            else:
                issue = "UNMATCHED_PEAR_PREDICTION"
                gt_name = CLASS_NAMES.get(best_gt["cls"], str(best_gt["cls"]))
                detail = "Pear prediction did not match a ground-truth pear at the configured IoU threshold."
            totals["false_positive_pear_boxes"] += 1
            image_rows.append({
                "image": str(image_path), "issue_type": issue,
                "gt_class": gt_name, "pred_class": "Pear",
                "confidence": round(pred["conf"], 4), "iou": round(best_iou, 4),
                "gt_box_xyxy": fmt_box(best_gt["xyxy"]) if best_gt else "",
                "pred_box_xyxy": fmt_box(pred["xyxy"]), "details": detail
            })

        rows.extend(image_rows)
        if image_rows:
            flagged_images.append((image_path, gt_boxes, pred_boxes, image_rows))
        if index % 50 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)} images; flagged images so far: {len(flagged_images)}")

    csv_path = out_dir / "pear_error_audit.csv"
    fields = ["image", "issue_type", "gt_class", "pred_class", "confidence", "iou",
              "gt_box_xyxy", "pred_box_xyxy", "details"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Save readable overlays. Green = ground truth, red = predictions.
    examples_to_save = flagged_images if args.max_examples == 0 else flagged_images[:args.max_examples]
    for image_path, gt_boxes, pred_boxes, image_rows in examples_to_save:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        canvas = image.copy()
        for gt in gt_boxes:
            draw_box(canvas, gt["xyxy"], (0, 200, 0), f"GT {CLASS_NAMES.get(gt['cls'], gt['cls'])}")
        for pred in pred_boxes:
            draw_box(canvas, pred["xyxy"], (0, 0, 255),
                     f"Pred {CLASS_NAMES.get(pred['cls'], pred['cls'])} {pred['conf']:.2f}")
        issue_types = sorted(set(r["issue_type"] for r in image_rows))
        header = "Issues: " + ", ".join(issue_types)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (30, 30, 30), -1)
        cv2.putText(canvas, header[:110], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in image_path.stem)
        output_path = examples_dir / f"{safe_stem}_pear_audit.jpg"
        cv2.imwrite(str(output_path), canvas)

    summary_path = out_dir / "pear_error_audit_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("PEAR-SPECIFIC YOLO VALIDATION ERROR AUDIT\n")
        f.write("=" * 44 + "\n")
        f.write(f"Images checked: {totals['images_checked']}\n")
        f.write(f"Ground-truth pear boxes: {totals['gt_pear_boxes']}\n")
        f.write(f"Ground-truth pear boxes matched by a predicted Pear: {totals['matched_pear_boxes']}\n")
        f.write(f"Missed pear boxes: {totals['missed_pear_boxes']}\n")
        f.write(f"Pear GT with wrong-class overlapping prediction: {totals['pear_class_confusions']}\n")
        f.write(f"Unmatched/false-positive pear predictions: {totals['false_positive_pear_boxes']}\n")
        f.write(f"Flagged images: {len(flagged_images)}\n")
        f.write(f"CSV report: {csv_path}\n")
        f.write(f"Annotated examples: {examples_dir}\n")
        f.write("\nInterpretation notes:\n")
        f.write("- This is a targeted IoU-based audit, not a replacement for Ultralytics mAP metrics.\n")
        f.write("- A class-confusion issue may also appear as a false-positive pear prediction.\n")
        f.write("- Inspect annotated images and source labels before deciding whether data relabeling is needed.\n")

    print("\nAUDIT COMPLETE")
    print(f"Images checked: {totals['images_checked']}")
    print(f"Ground-truth pear boxes: {totals['gt_pear_boxes']}")
    print(f"Matched pear boxes: {totals['matched_pear_boxes']}")
    print(f"Missed pears: {totals['missed_pear_boxes']}")
    print(f"Pear class confusions: {totals['pear_class_confusions']}")
    print(f"Unmatched/false-positive pear predictions: {totals['false_positive_pear_boxes']}")
    print(f"Flagged images: {len(flagged_images)}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Annotated examples: {examples_dir}")


if __name__ == "__main__":
    main()
