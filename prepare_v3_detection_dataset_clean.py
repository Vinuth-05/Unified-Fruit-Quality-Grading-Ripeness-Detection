from pathlib import Path
from collections import Counter
import shutil
import yaml

# Create a clean positive-image-only V3 detector dataset.
# Original Dataset is never modified.
# Detector classes: Apple=0, Orange=1, Pear=2.
# Only images with at least one valid supported-fruit box are copied.
# Train/valid/test split is preserved.

PROJECT_ROOT = Path(r"C:\Users\bdhul\Documents\Fruit_Grading_Project")
SOURCE_ROOT = PROJECT_ROOT / "Dataset"
OUTPUT_ROOT = PROJECT_ROOT / "Dataset_V3_Detection3Class_CLEAN"

SPLITS = {"train": "train", "valid": "valid", "test": "test"}
FRUIT_NAMES = {0: "Apple", 1: "Orange", 2: "Pear"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Original IDs from the source data.yaml.
ORIGINAL_TO_FRUIT = {}
for i in range(0, 4):
    ORIGINAL_TO_FRUIT[i] = 0
for i in range(20, 24):
    ORIGINAL_TO_FRUIT[i] = 1
for i in range(28, 32):
    ORIGINAL_TO_FRUIT[i] = 2


def parse_target_line(line):
    """
    Returns (target_class_id, xywh, was_polygon).
    Unsupported classes are filtered BEFORE coordinate validation.
    Supports standard YOLO boxes and polygon point lists.
    """
    parts = line.strip().split()
    if len(parts) < 1:
        raise ValueError("empty annotation")

    try:
        class_value = float(parts[0])
        original_id = int(class_value)
        if class_value != original_id:
            raise ValueError("class ID is not an integer")
    except ValueError as exc:
        raise ValueError(f"invalid class ID: {exc}") from exc

    # Skip all unsupported classes immediately. Their coordinates do not
    # affect this 3-class detector and should not be reported as target errors.
    if original_id not in ORIGINAL_TO_FRUIT:
        return None

    if len(parts) < 5:
        raise ValueError("target annotation has fewer than 5 fields")

    try:
        coords = [float(v) for v in parts[1:]]
    except ValueError as exc:
        raise ValueError("non-numeric coordinate") from exc

    if any(v < -1e-6 or v > 1.000001 for v in coords):
        raise ValueError("target coordinate outside normalized [0,1] range")

    if len(coords) == 4:
        x, y, w, h = coords
        if w <= 0 or h <= 0:
            raise ValueError("box width/height must be positive")
        if w > 1 or h > 1:
            raise ValueError("box width/height exceeds 1")
        # Permit a center slightly outside [0,1] only if within tiny tolerance,
        # then clamp center to the valid range. Larger errors are reported above.
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        return ORIGINAL_TO_FRUIT[original_id], (x, y, w, h), False

    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError("polygon must contain at least 3 x/y coordinate pairs")

    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w, h = xmax - xmin, ymax - ymin
    if w <= 0 or h <= 0:
        raise ValueError("polygon produces a zero-area box")
    return ORIGINAL_TO_FRUIT[original_id], (
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        w,
        h,
    ), True


def main():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_ROOT}")

    if OUTPUT_ROOT.exists():
        answer = input(
            f"Output already exists:\n{OUTPUT_ROOT}\n"
            "Type OVERWRITE to delete and rebuild ONLY this output folder: "
        ).strip()
        if answer != "OVERWRITE":
            print("Cancelled. Nothing was changed.")
            return
        shutil.rmtree(OUTPUT_ROOT)

    totals = Counter()
    split_counts = {}
    issues = []

    for split, source_split in SPLITS.items():
        image_dir = SOURCE_ROOT / source_split / "images"
        label_dir = SOURCE_ROOT / source_split / "labels"
        out_image_dir = OUTPUT_ROOT / split / "images"
        out_label_dir = OUTPUT_ROOT / split / "labels"
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            issues.append(f"{split}: missing image directory: {image_dir}")
            split_counts[split] = Counter()
            continue

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        source_stems = {p.stem for p in images}
        counts = Counter()

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                totals["missing_label_files"] += 1
                continue

            output_lines = []
            if label_path.exists():
                for line_no, line in enumerate(
                    label_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                ):
                    if not line.strip():
                        continue
                    try:
                        parsed = parse_target_line(line)
                    except ValueError as exc:
                        totals["invalid_target_lines"] += 1
                        issues.append(
                            f"{split}/{label_path.name}:{line_no}: {exc}; "
                            f"line={line[:140]}"
                        )
                        continue

                    if parsed is None:
                        totals["unsupported_instances_skipped"] += 1
                        continue

                    class_id, (x, y, w, h), was_polygon = parsed
                    output_lines.append(
                        f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                    )
                    counts[class_id] += 1
                    totals["boxes_written"] += 1
                    if was_polygon:
                        totals["polygons_converted"] += 1
                    else:
                        totals["boxes_preserved"] += 1

            # Positive-image-only: copy only if at least one supported box exists.
            if output_lines:
                shutil.copy2(image_path, out_image_dir / image_path.name)
                (out_label_dir / f"{image_path.stem}.txt").write_text(
                    "\n".join(output_lines) + "\n", encoding="utf-8"
                )
                counts["positive_images"] += 1
                totals["positive_images_copied"] += 1
            else:
                totals["images_without_supported_boxes_excluded"] += 1

        if label_dir.exists():
            for label_path in label_dir.glob("*.txt"):
                if label_path.stem not in source_stems:
                    totals["orphan_label_files"] += 1
                    issues.append(f"{split}: label has no matching image: {label_path.name}")

        split_counts[split] = counts

    yaml_data = {
        "path": str(OUTPUT_ROOT).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 3,
        "names": {0: "Apple", 1: "Orange", 2: "Pear"},
    }
    (OUTPUT_ROOT / "data.yaml").write_text(
        yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    report = [
        "V3 CLEAN DETECTION DATASET REPORT",
        f"Source (untouched): {SOURCE_ROOT}",
        f"Output: {OUTPUT_ROOT}",
        "Strategy: positive-image-only; images with no valid Apple/Orange/Pear boxes excluded.",
        "Train/valid/test split preserved. Unsupported fruit classes are ignored before coordinate checks.",
        "",
    ]
    for split in SPLITS:
        c = split_counts.get(split, Counter())
        report += [
            f"{split}:",
            f"  positive images copied: {c['positive_images']}",
            f"  Apple boxes: {c[0]}",
            f"  Orange boxes: {c[1]}",
            f"  Pear boxes: {c[2]}",
            "",
        ]
    report += [
        "Overall:",
        f"  positive images copied: {totals['positive_images_copied']}",
        f"  boxes written: {totals['boxes_written']}",
        f"  polygons converted: {totals['polygons_converted']}",
        f"  ordinary boxes preserved: {totals['boxes_preserved']}",
        f"  unsupported instances skipped: {totals['unsupported_instances_skipped']}",
        f"  invalid target annotation lines: {totals['invalid_target_lines']}",
        f"  images without supported boxes excluded: {totals['images_without_supported_boxes_excluded']}",
        f"  missing source label files: {totals['missing_label_files']}",
        f"  orphan label files: {totals['orphan_label_files']}",
        "",
        "Issues to review:",
    ]
    report.extend(f"  - {issue}" for issue in issues)
    report_path = OUTPUT_ROOT / "preparation_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nCreated YAML: {OUTPUT_ROOT / 'data.yaml'}")
    print(f"Created report: {report_path}")
    print("\nReview the counts and invalid TARGET annotation lines before training.")


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        raise SystemExit("Install PyYAML first: python -m pip install pyyaml")
    main()
