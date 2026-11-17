"""
One-to-one Pear detection audit for YOLO validation/test data.
Does not train, alter, or overwrite model weights.

Run from backend:
python audit_yolo_pear_errors_final.py --weights models/best.pt --data "C:/Users/bdhul/Documents/Fruit_Grading_Project/Dataset_V3_Detection3Class_CLEAN/data.yaml"
"""
import argparse
import csv
import os
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

CLASS_NAMES = {0: "Apple", 1: "Orange", 2: "Pear"}
PEAR_ID = 2
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    ap = argparse.ArgumentParser(description="One-to-one Pear-specific YOLO error audit")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--data", default="C:/Users/bdhul/Documents/Fruit_Grading_Project/Dataset_V3_Detection3Class_CLEAN/data.yaml")
    ap.add_argument("--split", choices=["val", "valid", "test"], default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--match-iou", type=float, default=0.50)
    ap.add_argument("--max-examples", type=int, default=60, help="0 saves all flagged images")
    ap.add_argument("--out", default="pear_error_audit_final")
    return ap.parse_args()


def dataset_root(yaml_path, cfg):
    yaml_dir = yaml_path.parent
    raw = cfg.get("path")
    if raw:
        candidate = Path(str(raw))
        if candidate.exists():
            return candidate.resolve()
        # Handle Windows-style paths stored in YAML, and data.yaml copied to another OS.
        candidate = Path(str(raw).replace("\\", os.sep).replace("/", os.sep))
        if candidate.exists():
            return candidate.resolve()
    for candidate in (yaml_dir, yaml_dir.parent):
        if any((candidate / split).exists() for split in ("train", "valid", "val", "test")):
            return candidate.resolve()
    raise FileNotFoundError("Cannot resolve dataset root. Check the 'path' field in data.yaml.")


def resolve_split_dir(root, yaml_dir, value):
    p = Path(str(value))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates += [root / p, yaml_dir / p]
    # If a path includes a stale absolute prefix, retain the split/images suffix.
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.lower() in ("train", "valid", "val", "test"):
            candidates.append(root.joinpath(*parts[i:]))
            candidates.append(root.joinpath(*parts[i:]).parent)
    for candidate in candidates:
        if candidate.exists():
            candidate = candidate.resolve()
            if candidate.name.lower() == "images":
                return candidate
            if (candidate / "images").is_dir():
                return (candidate / "images").resolve()
    raise FileNotFoundError(f"Could not find image folder for split entry: {value}")


def get_dirs(root, yaml_path, cfg, requested):
    keys = [requested]
    if requested == "val":
        keys.append("valid")
    if requested == "valid":
        keys.append("val")
    value = next((cfg[k] for k in keys if cfg.get(k)), None)
    if value is None:
        raise KeyError(f"data.yaml has no split entry for {requested}/valid.")
    image_dir = resolve_split_dir(root, yaml_path.parent, value)
    label_dir = image_dir.parent / "labels"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {label_dir}")
    return image_dir, label_dir


def read_labels(path, w, h):
    out = []
    if not path.exists():
        return out
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        # YOLO normalized xywh -> original-image pixel xyxy.
        box = [(xc - bw / 2) * w, (yc - bh / 2) * h,
               (xc + bw / 2) * w, (yc + bh / 2) * h]
        box[0] = max(0.0, min(float(w), box[0]))
        box[2] = max(0.0, min(float(w), box[2]))
        box[1] = max(0.0, min(float(h), box[1]))
        box[3] = max(0.0, min(float(h), box[3]))
        out.append({"cls": cls, "box": box, "line": line_no})
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def fmt(box):
    return ",".join(f"{float(v):.1f}" for v in box)


def draw(image, box, color, label):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (max(2, x1), max(18, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    yaml_path = Path(args.data).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    examples = out / "annotated_examples"
    examples.mkdir(parents=True, exist_ok=True)

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    root = dataset_root(yaml_path, cfg)
    image_dir, label_dir = get_dirs(root, yaml_path, cfg, args.split)
    images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"Dataset root : {root}")
    print(f"Split images : {image_dir}")
    print(f"Split labels : {label_dir}")
    print(f"Images       : {len(images)}")
    print(f"Weights      : {weights}")
    print(f"Settings     : conf={args.conf}, NMS IoU={args.nms_iou}, match IoU={args.match_iou}")
    print("Matching     : greedy one-to-one, class-aware first")

    model = YOLO(str(weights))
    rows, flagged = [], []
    stats = {"images_checked": 0, "gt_pear": 0, "matched_pear": 0,
             "missed_pear": 0, "pear_class_confusions": 0,
             "unmatched_pear_predictions": 0, "missing_label_files": 0}

    for n, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Cannot read image: {image_path}")
            continue
        h, w = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            stats["missing_label_files"] += 1
        gt = read_labels(label_path, w, h)
        result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf,
                               iou=args.nms_iou, verbose=False, save=False)[0]
        pred = []
        if result.boxes is not None:
            for b, c, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy().astype(int),
                                  result.boxes.conf.cpu().numpy()):
                pred.append({"cls": int(c), "conf": float(conf), "box": [float(v) for v in b]})

        stats["images_checked"] += 1
        pears_gt = [g for g in gt if g["cls"] == PEAR_ID]
        pear_pred_indices = [i for i, p in enumerate(pred) if p["cls"] == PEAR_ID]
        stats["gt_pear"] += len(pears_gt)
        image_rows = []

        # Global greedy one-to-one matching among same-class boxes.
        candidates = []
        for gi, g in enumerate(gt):
            for pi, p in enumerate(pred):
                if g["cls"] == p["cls"]:
                    score = iou(g["box"], p["box"])
                    if score >= args.match_iou:
                        candidates.append((score, gi, pi))
        candidates.sort(reverse=True, key=lambda x: x[0])
        matched_g, matched_p = set(), set()
        for score, gi, pi in candidates:
            if gi not in matched_g and pi not in matched_p:
                matched_g.add(gi)
                matched_p.add(pi)

        # Unmatched ground-truth Pear: class confusion if wrong class overlaps; else missed.
        for gi, g in enumerate(gt):
            if g["cls"] != PEAR_ID or gi in matched_g:
                continue
            wrong = [(iou(g["box"], p["box"]), pi, p) for pi, p in enumerate(pred)
                     if p["cls"] != PEAR_ID and iou(g["box"], p["box"]) >= args.match_iou]
            if wrong:
                score, pi, p = max(wrong, key=lambda x: x[0])
                issue = "PEAR_GT_WRONG_CLASS_PREDICTION"
                stats["pear_class_confusions"] += 1
                pred_name = CLASS_NAMES.get(p["cls"], str(p["cls"]))
                conf = p["conf"]
                pred_box = fmt(p["box"])
                detail = "Pear GT has an overlapping prediction with a different class."
            else:
                issue = "MISSED_PEAR"
                stats["missed_pear"] += 1
                best = max((iou(g["box"], p["box"]) for p in pred), default=0.0)
                score, pred_name, conf, pred_box = best, "None", "", ""
                detail = "No same-class Pear prediction matched this GT one-to-one; no wrong-class box met the IoU threshold."
            image_rows.append({"image": str(image_path), "issue_type": issue,
                               "gt_class": "Pear", "pred_class": pred_name,
                               "confidence": round(conf, 4) if conf != "" else "",
                               "iou": round(score, 4), "gt_box_xyxy": fmt(g["box"]),
                               "pred_box_xyxy": pred_box, "details": detail})

        # Every unmatched predicted Pear is reported once, even if it overlaps a GT Pear
        # already matched by another prediction. Such cases can reveal duplicate boxes.
        for pi in pear_pred_indices:
            if pi in matched_p:
                continue
            p = pred[pi]
            best_gt_iou, best_gt = max(
                ((iou(p["box"], g["box"]), g) for g in gt),
                key=lambda x: x[0], default=(0.0, None))
            if best_gt is None:
                issue, gt_name = "UNMATCHED_PEAR_PREDICTION_NO_GT", "None"
            elif best_gt["cls"] == PEAR_ID and best_gt_iou >= args.match_iou:
                issue, gt_name = "POSSIBLE_DUPLICATE_PEAR_PREDICTION", "Pear"
            elif best_gt["cls"] != PEAR_ID and best_gt_iou >= args.match_iou:
                issue, gt_name = "WRONG_CLASS_PEAR_ON_OTHER_FRUIT", CLASS_NAMES.get(best_gt["cls"], str(best_gt["cls"]))
            else:
                issue, gt_name = "UNMATCHED_PEAR_PREDICTION", CLASS_NAMES.get(best_gt["cls"], str(best_gt["cls"])) if best_gt else "None"
            stats["unmatched_pear_predictions"] += 1
            image_rows.append({"image": str(image_path), "issue_type": issue,
                               "gt_class": gt_name, "pred_class": "Pear",
                               "confidence": round(p["conf"], 4), "iou": round(best_gt_iou, 4),
                               "gt_box_xyxy": fmt(best_gt["box"]) if best_gt else "",
                               "pred_box_xyxy": fmt(p["box"]),
                               "details": "Predicted Pear box was not assigned to a same-class GT box in one-to-one matching."})

        rows.extend(image_rows)
        if image_rows:
            flagged.append((image_path, gt, pred, image_rows))
        if n % 50 == 0 or n == len(images):
            print(f"Processed {n}/{len(images)} images; flagged images: {len(flagged)}")

    fields = ["image", "issue_type", "gt_class", "pred_class", "confidence", "iou",
              "gt_box_xyxy", "pred_box_xyxy", "details"]
    csv_path = out / "pear_error_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    chosen = flagged if args.max_examples == 0 else flagged[:args.max_examples]
    for image_path, gt, pred, issue_rows in chosen:
        canvas = cv2.imread(str(image_path))
        if canvas is None:
            continue
        for g in gt:
            draw(canvas, g["box"], (0, 200, 0), f"GT {CLASS_NAMES.get(g['cls'], g['cls'])}")
        for p in pred:
            draw(canvas, p["box"], (0, 0, 255), f"Pred {CLASS_NAMES.get(p['cls'], p['cls'])} {p['conf']:.2f}")
        issue_text = ", ".join(sorted({r["issue_type"] for r in issue_rows}))
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (30, 30, 30), -1)
        cv2.putText(canvas, issue_text[:120], (7, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in image_path.stem)
        cv2.imwrite(str(examples / f"{safe}_pear_audit.jpg"), canvas)

    summary = out / "pear_error_audit_summary.txt"

    summary = out / "pear_error_audit_summary.txt"

    # Recompute matched pear GT from rows so total is correct across all images.
    missed = stats["missed_pear"] + stats["pear_class_confusions"]
    matched_pear = max(0, stats["gt_pear"] - missed)
    with summary.open("w", encoding="utf-8") as f:
        f.write("ONE-TO-ONE PEAR ERROR AUDIT\n" + "=" * 32 + "\n")
        f.write(f"Images checked: {stats['images_checked']}\n")
        f.write(f"Ground-truth Pear boxes: {stats['gt_pear']}\n")
        f.write(f"Matched Pear boxes (same class, one-to-one): {matched_pear}\n")
        f.write(f"Missed Pear boxes: {stats['missed_pear']}\n")
        f.write(f"Pear class confusions: {stats['pear_class_confusions']}\n")
        f.write(f"Unmatched Pear predictions: {stats['unmatched_pear_predictions']}\n")
        f.write(f"Images with missing label files: {stats['missing_label_files']}\n")
        f.write(f"Flagged images: {len(flagged)}\n")
        f.write(f"CSV: {csv_path}\nAnnotated examples: {examples}\n")
        f.write("Note: diagnostic audit only, not official precision/recall/mAP.\n")

    print("\nAUDIT COMPLETE")
    print(f"Images checked: {stats['images_checked']}")
    print(f"Ground-truth Pear boxes: {stats['gt_pear']}")
    print(f"Matched Pear boxes (one-to-one): {matched_pear}")
    print(f"Missed Pear boxes: {stats['missed_pear']}")
    print(f"Pear class confusions: {stats['pear_class_confusions']}")
    print(f"Unmatched Pear predictions: {stats['unmatched_pear_predictions']}")
    print(f"Flagged images: {len(flagged)}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary}")
    print(f"Annotated examples: {examples}")


if __name__ == "__main__":
    main()
