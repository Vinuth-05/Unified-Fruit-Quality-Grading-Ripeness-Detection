"""
YOLO V3 DATASET LABEL AUDIT + CONTACT SHEET
------------------------------------------
Read-only diagnostic: does not edit images, labels, dataset, or model.

Run from your backend folder:
    python audit_yolo_dataset.py

Optional:
    python audit_yolo_dataset.py --samples 24 --split train --seed 42

It samples images from the CLEAN dataset, draws their YOLO labels, and reports:
- missing label files
- malformed label rows
- invalid class IDs / normalized coordinates
- very small or suspicious boxes
- class distribution in the sampled images

Default dataset path is the CLEAN V3 dataset in the project.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle


DEFAULT_DATASET = Path(
    r"C:\Users\bdhul\Documents\Fruit_Grading_Project\Dataset_V3_Detection3Class_CLEAN"
)
CLASS_NAMES = {0: "Apple", 1: "Orange", 2: "Pear"}
CLASS_COLORS = {
    0: "lime",
    1: "orange",
    2: "deepskyblue",
}


def find_label(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(images_dir)
    return (labels_dir / relative).with_suffix(".txt")


def parse_label_file(label_path: Path, image_width: int, image_height: int):
    rows = []
    issues = []
    if not label_path.exists():
        return rows, [f"Missing label file: {label_path.name}"]

    for line_no, raw_line in enumerate(
        label_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(
                f"{label_path.name}:{line_no}: expected 5 values, found {len(parts)}"
            )
            continue
        try:
            class_value = float(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(f"{label_path.name}:{line_no}: non-numeric value")
            continue

        class_id = int(class_value)
        if class_value != class_id or class_id not in CLASS_NAMES:
            issues.append(
                f"{label_path.name}:{line_no}: invalid class id {parts[0]}"
            )
            continue

        xc, yc, w, h = coords
        if not all(0 <= v <= 1 for v in coords):
            issues.append(
                f"{label_path.name}:{line_no}: normalized values outside [0,1]"
            )
            continue
        if w <= 0 or h <= 0:
            issues.append(f"{label_path.name}:{line_no}: non-positive width/height")
            continue

        x1 = (xc - w / 2) * image_width
        y1 = (yc - h / 2) * image_height
        x2 = (xc + w / 2) * image_width
        y2 = (yc + h / 2) * image_height

        # Flag, but still draw, boxes that extend beyond the image boundary.
        if x1 < -1 or y1 < -1 or x2 > image_width + 1 or y2 > image_height + 1:
            issues.append(
                f"{label_path.name}:{line_no}: box extends beyond image boundary"
            )
        if w < 0.02 or h < 0.02:
            issues.append(
                f"{label_path.name}:{line_no}: very small box (w={w:.4f}, h={h:.4f})"
            )

        rows.append(
            {
                "class_id": class_id,
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
                "xyxy": (x1, y1, x2, y2),
                "line_no": line_no,
            }
        )
    return rows, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    images_dir = dataset / args.split / "images"
    labels_dir = dataset / args.split / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )
    if not all_images:
        raise RuntimeError(f"No images found in {images_dir}")

    rng = random.Random(args.seed)
    selected = all_images if len(all_images) <= args.samples else rng.sample(
        all_images, args.samples
    )
    selected = sorted(selected)

    cols = 4
    rows_count = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(18, 4.5 * rows_count))
    if hasattr(axes, "flat"):
        axes = list(axes.flat)
    else:
        axes = [axes]

    sampled_counts = Counter()
    issues_found = []
    missing_labels = 0
    total_boxes = 0

    for ax, image_path in zip(axes, selected):
        label_path = find_label(image_path, images_dir, labels_dir)
        try:
            with Image.open(image_path) as src:
                image = src.convert("RGB")
            width, height = image.size
            label_rows, issues = parse_label_file(label_path, width, height)
            issues_found.extend(
                f"{image_path.relative_to(images_dir)}: {issue}" for issue in issues
            )
            if not label_path.exists():
                missing_labels += 1

            ax.imshow(image)
            for item in label_rows:
                class_id = item["class_id"]
                x1, y1, x2, y2 = item["xyxy"]
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=CLASS_COLORS[class_id], facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(
                    max(0, x1), max(0, y1 - 2),
                    CLASS_NAMES[class_id],
                    color="black", fontsize=8, weight="bold",
                    bbox={"facecolor": CLASS_COLORS[class_id], "alpha": 0.85, "pad": 1}
                )
                sampled_counts[class_id] += 1
                total_boxes += 1

            ax.set_title(f"{image_path.name}\n{len(label_rows)} boxes", fontsize=8)
            ax.axis("off")
        except Exception as exc:
            ax.set_title(f"{image_path.name}\nERROR: {exc}", fontsize=8)
            ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(
        f"YOLO Dataset Audit | split={args.split} | samples={len(selected)}\n"
        "Green=Apple | Orange=Orange | Blue=Pear",
        fontsize=15
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_image = dataset / f"label_audit_{args.split}_{len(selected)}samples.png"
    output_report = dataset / f"label_audit_{args.split}_{len(selected)}samples.txt"
    fig.savefig(output_image, dpi=140, bbox_inches="tight")
    plt.close(fig)

    report_lines = [
        "YOLO DATASET LABEL AUDIT",
        f"Dataset: {dataset}",
        f"Split: {args.split}",
        f"Images available in split: {len(all_images)}",
        f"Images sampled: {len(selected)}",
        f"Sampled boxes: {total_boxes}",
        f"Sampled Apple boxes: {sampled_counts[0]}",
        f"Sampled Orange boxes: {sampled_counts[1]}",
        f"Sampled Pear boxes: {sampled_counts[2]}",
        f"Sampled images missing label files: {missing_labels}",
        f"Issue count in sampled labels: {len(issues_found)}",
        "",
        "ISSUES:",
    ]
    report_lines.extend(issues_found if issues_found else ["No structural issues found in sampled labels."])
    output_report.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nContact sheet saved to: {output_image}")
    print(f"Audit report saved to: {output_report}")
    print("\nOpen the contact sheet and visually verify that each box tightly encloses")
    print("the fruit named by its label. Structural checks cannot detect a wrong class label.")


if __name__ == "__main__":
    main()
