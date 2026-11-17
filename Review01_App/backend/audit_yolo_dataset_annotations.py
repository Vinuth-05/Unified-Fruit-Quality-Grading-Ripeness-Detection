
"""
YOLO Dataset Annotation Audit
Classes: 0 = Apple, 1 = Orange, 2 = Pear

Checks:
- Missing and empty label files
- Invalid rows and class IDs
- Out-of-range normalized coordinates
- Zero/negative-size boxes
- Very large and very small boxes
- Overlapping ground-truth boxes
- Unusually large Pear boxes
- Saves CSV, summary, and annotated suspicious examples
"""

import argparse
import csv
import math
from pathlib import Path

import cv2

CLASS_NAMES = {0: "Apple", 1: "Orange", 2: "Pear"}

# Allow tiny decimal-rounding differences in annotation coordinates.
TOLERANCE = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit YOLO dataset annotations."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            r"C:\Users\bdhul\Documents\Fruit_Grading_Project"
            r"\Dataset_V3_Detection3Class_CLEAN"
        ),
    )
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--large-area", type=float, default=0.80)
    parser.add_argument("--small-area", type=float, default=0.001)
    parser.add_argument("--pear-large-area", type=float, default=0.80)
    parser.add_argument("--overlap-iou", type=float, default=0.50)
    parser.add_argument("--max-examples", type=int, default=100)
    return parser.parse_args()


def find_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def get_label_path(image_path, images_dir, labels_dir):
    relative = image_path.relative_to(images_dir)
    return (labels_dir / relative).with_suffix(".txt")


def parse_yolo_row(line):
    parts = line.strip().split()

    if len(parts) != 5:
        raise ValueError(
            "Expected 5 values: class_id xc yc width height"
        )

    try:
        class_value = float(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError("Non-numeric value found.") from exc

    if not math.isfinite(class_value):
        raise ValueError("Invalid class ID.")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("NaN or infinite coordinate found.")
    if not class_value.is_integer():
        raise ValueError("Class ID must be an integer.")

    return int(class_value), values


def box_edges(xc, yc, width, height):
    left = xc - width / 2
    top = yc - height / 2
    right = xc + width / 2
    bottom = yc + height / 2
    return left, top, right, bottom


def is_out_of_range(xc, yc, width, height):
    # YOLO center, width, and height must be normalized.
    values = (xc, yc, width, height)

    if any(
        value < -TOLERANCE or value > 1.0 + TOLERANCE
        for value in values
    ):
        return True

    left, top, right, bottom = box_edges(
        xc, yc, width, height
    )

    # Allow tiny rounding differences such as -0.0000005.
    return (
        left < -TOLERANCE
        or top < -TOLERANCE
        or right > 1.0 + TOLERANCE
        or bottom > 1.0 + TOLERANCE
    )


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0.0, x2 - x1) * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )
    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def add_issue(rows, flagged, image_path, issue_type,
              class_id="", class_name="", line_no="",
              coords="", area="", overlap="", other_line="",
              details=""):

    rows.append({
        "image": str(image_path),
        "issue_type": issue_type,
        "class_id": class_id,
        "class_name": class_name,
        "label_line": line_no,
        "box_xywh_normalized": coords,
        "area_fraction": area,
        "overlap_iou": overlap,
        "other_label_line": other_line,
        "details": details,
    })

    flagged.setdefault(image_path, set()).add(issue_type)


def draw_box(image, box, label, color):
    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(0, min(image_width - 1, round(x1)))
    x2 = max(0, min(image_width - 1, round(x2)))
    y1 = max(0, min(image_height - 1, round(y1)))
    y2 = max(0, min(image_height - 1, round(y2)))

    cv2.rectangle(
        image, (x1, y1), (x2, y2), color, 2
    )
    cv2.putText(
        image,
        label,
        (max(2, x1), max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def save_annotated_example(image_path, label_path, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return

    image_height, image_width = image.shape[:2]

    if label_path.exists():
        lines = label_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()

        for line in lines:
            if not line.strip():
                continue

            try:
                class_id, values = parse_yolo_row(line)
            except ValueError:
                continue

            xc, yc, width, height = values
            left, top, right, bottom = box_edges(
                xc, yc, width, height
            )

            pixel_box = (
                left * image_width,
                top * image_height,
                right * image_width,
                bottom * image_height,
            )

            color = (
                (0, 165, 255)
                if class_id == 2
                else (0, 255, 0)
            )

            draw_box(
                image,
                pixel_box,
                CLASS_NAMES.get(class_id, f"Class {class_id}"),
                color,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main():
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    split_dir = dataset_root / args.split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else Path(__file__).resolve().parent
        / "annotation_diagnostics"
    )

    examples_dir = output_dir / "suspicious_examples"
    csv_path = output_dir / "annotation_diagnostics.csv"
    summary_path = (
        output_dir / "annotation_diagnostics_summary.txt"
    )

    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    images = find_images(images_dir)

    if not images:
        raise SystemExit(f"No images found in: {images_dir}")

    totals = {
        "images_checked": 0,
        "images_without_label_file": 0,
        "empty_label_files": 0,
        "invalid_rows": 0,
        "invalid_class_ids": 0,
        "out_of_range_boxes": 0,
        "zero_or_negative_boxes": 0,
        "large_boxes": 0,
        "small_boxes": 0,
        "high_overlap_pairs": 0,
        "pear_large_boxes": 0,
        "total_gt_boxes": 0,
    }

    rows = []
    flagged = {}

    for index, image_path in enumerate(images, start=1):
        totals["images_checked"] += 1

        label_path = get_label_path(
            image_path, images_dir, labels_dir
        )

        if not label_path.exists():
            totals["images_without_label_file"] += 1

            add_issue(
                rows, flagged, image_path,
                "MISSING_LABEL_FILE",
                details=f"Label file not found: {label_path}",
            )

            continue

        content = label_path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not content:
            totals["empty_label_files"] += 1

            add_issue(
                rows, flagged, image_path,
                "EMPTY_LABEL_FILE",
                details="Label file exists but is empty.",
            )

            continue

        boxes = []

        for line_no, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue

            try:
                class_id, values = parse_yolo_row(line)
            except ValueError as exc:
                totals["invalid_rows"] += 1

                add_issue(
                    rows, flagged, image_path,
                    "INVALID_ROW",
                    line_no=line_no,
                    coords=line.strip(),
                    details=str(exc),
                )
                continue

            xc, yc, width, height = values
            coords = " ".join(str(value) for value in values)
            area = width * height
            class_name = CLASS_NAMES.get(class_id, "Unknown")

            totals["total_gt_boxes"] += 1

            if class_id not in CLASS_NAMES:
                totals["invalid_class_ids"] += 1

                add_issue(
                    rows, flagged, image_path,
                    "INVALID_CLASS_ID",
                    class_id, class_name, line_no,
                    coords, round(area, 6),
                    details=(
                        "Class ID must be 0 (Apple), "
                        "1 (Orange), or 2 (Pear)."
                    ),
                )

            if is_out_of_range(xc, yc, width, height):
                totals["out_of_range_boxes"] += 1

                add_issue(
                    rows, flagged, image_path,
                    "OUT_OF_RANGE_BOX",
                    class_id, class_name, line_no,
                    coords, round(area, 6),
                    details=(
                        "Normalized box or its edges extend "
                        "outside [0,1] beyond rounding tolerance."
                    ),
                )

            if width <= 0 or height <= 0:
                totals["zero_or_negative_boxes"] += 1

                add_issue(
                    rows, flagged, image_path,
                    "ZERO_OR_NEGATIVE_BOX",
                    class_id, class_name, line_no,
                    coords, round(area, 6),
                    details="Width and height must both be positive.",
                )

            if width > 0 and height > 0:
                if area >= args.large_area:
                    totals["large_boxes"] += 1

                    add_issue(
                        rows, flagged, image_path,
                        "VERY_LARGE_BOX",
                        class_id, class_name, line_no,
                        coords, round(area, 6),
                        details=(
                            f"Box area {area:.4f} is at least "
                            f"{args.large_area:.4f}; inspect visually."
                        ),
                    )

                if (
                    class_id == 2
                    and area >= args.pear_large_area
                ):
                    totals["pear_large_boxes"] += 1

                    add_issue(
                        rows, flagged, image_path,
                        "VERY_LARGE_PEAR_BOX",
                        class_id, "Pear", line_no,
                        coords, round(area, 6),
                        details="Unusually large Pear box; inspect visually.",
                    )

                if area <= args.small_area:
                    totals["small_boxes"] += 1

                    add_issue(
                        rows, flagged, image_path,
                        "VERY_SMALL_BOX",
                        class_id, class_name, line_no,
                        coords, round(area, 6),
                        details=(
                            f"Box area {area:.6f} is at most "
                            f"{args.small_area:.6f}; inspect visually."
                        ),
                    )

            left, top, right, bottom = box_edges(
                xc, yc, width, height
            )

            boxes.append({
                "line": line_no,
                "class_id": class_id,
                "class_name": class_name,
                "coords": coords,
                "area": area,
                "box": (left, top, right, bottom),
            })

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                overlap = calculate_iou(a["box"], b["box"])

                if overlap >= args.overlap_iou:
                    totals["high_overlap_pairs"] += 1

                    add_issue(
                        rows, flagged, image_path,
                        "HIGH_OVERLAP_GT_BOXES",
                        a["class_id"], a["class_name"],
                        a["line"], a["coords"],
                        round(a["area"], 6),
                        round(overlap, 6),
                        b["line"],
                        (
                            f"Ground-truth lines {a['line']} and "
                            f"{b['line']} have IoU {overlap:.3f}."
                        ),
                    )

        if index % 100 == 0 or index == len(images):
            print(
                f"Checked {index}/{len(images)} images; "
                f"suspicious images: {len(flagged)}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image", "issue_type", "class_id", "class_name",
        "label_line", "box_xywh_normalized", "area_fraction",
        "overlap_iou", "other_label_line", "details",
    ]

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    for image_path in sorted(flagged, key=str)[:args.max_examples]:
        label_path = get_label_path(
            image_path, images_dir, labels_dir
        )

        issue_names = "_".join(sorted(flagged[image_path]))
        filename = f"{image_path.stem[:90]}__{issue_names[:70]}.jpg"

        save_annotated_example(
            image_path,
            label_path,
            examples_dir / filename,
        )

    summary = [
        "YOLO ANNOTATION DIAGNOSTICS SUMMARY",
        f"Dataset root: {dataset_root}",
        f"Split: {args.split}",
        f"Coordinate rounding tolerance: {TOLERANCE}",
        "",
        f"Images checked: {totals['images_checked']}",
        f"Total valid-format GT boxes: {totals['total_gt_boxes']}",
        f"Images missing label files: {totals['images_without_label_file']}",
        f"Empty label files: {totals['empty_label_files']}",
        f"Invalid rows: {totals['invalid_rows']}",
        f"Invalid class IDs: {totals['invalid_class_ids']}",
        f"Out-of-range boxes: {totals['out_of_range_boxes']}",
        f"Zero/negative boxes: {totals['zero_or_negative_boxes']}",
        f"Very large boxes: {totals['large_boxes']} "
        f"(Pear: {totals['pear_large_boxes']})",
        f"Very small boxes: {totals['small_boxes']}",
        f"High-overlap GT box pairs: {totals['high_overlap_pairs']}",
        f"Flagged images: {len(flagged)}",
        "",
        "Flags indicate items to inspect, not automatically incorrect labels.",
        "Tiny rounding discrepancies within tolerance are not flagged.",
        f"CSV: {csv_path}",
        f"Summary: {summary_path}",
        f"Annotated examples: {examples_dir}",
    ]

    summary_path.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("\nANNOTATION DIAGNOSTICS COMPLETE")

    for label, key in [
        ("Images checked", "images_checked"),
        ("Total valid-format GT boxes", "total_gt_boxes"),
        ("Images missing label files", "images_without_label_file"),
        ("Empty label files", "empty_label_files"),
        ("Invalid rows", "invalid_rows"),
        ("Invalid class IDs", "invalid_class_ids"),
        ("Out-of-range boxes", "out_of_range_boxes"),
        ("Zero/negative boxes", "zero_or_negative_boxes"),
        ("Very small boxes", "small_boxes"),
        ("High-overlap GT box pairs", "high_overlap_pairs"),
    ]:
        print(f"{label}: {totals[key]}")

    print(
        f"Very large boxes: {totals['large_boxes']} "
        f"(Pear: {totals['pear_large_boxes']})"
    )
    print(f"Flagged images: {len(flagged)}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Annotated examples: {examples_dir}")


if __name__ == "__main__":
    main()
