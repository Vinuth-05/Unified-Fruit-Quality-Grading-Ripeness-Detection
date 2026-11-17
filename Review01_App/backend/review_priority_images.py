#!/usr/bin/env python3
"""
Create a small, prioritized visual review set from annotation_diagnostics.csv.

Run from the backend folder:
    python review_priority_images.py

This script only reads the diagnostic CSV and source images. It does not modify
dataset labels, model weights, or the original images.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError:
    print("Missing Pillow. Install it with: python -m pip install pillow")
    raise SystemExit(1)


DEFAULT_CSV = Path("annotation_diagnostics") / "annotation_diagnostics.csv"
DEFAULT_OUT = Path("annotation_diagnostics") / "priority_review"

# Preferred order: issues most useful to inspect first.
PRIORITY_TERMS = [
    ("VERY_LARGE_PEAR_BOX", 4),
    ("OUT_OF_RANGE_BOX", 6),
    ("VERY_LARGE_BOX", 4),
    ("VERY_SMALL_BOX", 3),
    ("HIGH_OVERLAP", 3),
]


def first_value(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def parse_box(value: str) -> tuple[float, float, float, float] | None:
    """Parse a YOLO normalized xywh box from common CSV representations."""
    if not value:
        return None
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", value)
    if len(nums) < 4:
        return None
    try:
        return tuple(float(x) for x in nums[:4])  # type: ignore[return-value]
    except ValueError:
        return None


def parse_flags(row: dict[str, str]) -> list[str]:
    raw = first_value(row, ("flags", "flag", "issue", "issues", "diagnostic", "reason", "problem", "category"))
    if raw:
        parts = re.split(r"[;|,]+", raw)
        return [x.strip() for x in parts if x.strip()]
    # Some CSVs have one flag per row, with its name in a dedicated column.
    for key, value in row.items():
        if key and value and any(term in key.lower() for term in ("flag", "issue", "diagnostic", "problem")):
            return [str(value).strip()]
    return ["FLAGGED"]


def image_path_from_row(row: dict[str, str]) -> str:
    return first_value(row, ("image_path", "image", "filepath", "file_path", "path", "filename", "file"))


def area_of_box(box: tuple[float, float, float, float] | None) -> float:
    if not box:
        return 0.0
    _, _, w, h = box
    return abs(w * h)


def select_rows(rows: list[dict[str, str]], per_category: int) -> list[dict[str, Any]]:
    # Deduplicate rows by image, aggregating all flags/boxes for each image.
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        img_path = image_path_from_row(row)
        if not img_path:
            continue
        key = img_path.strip().strip('"')
        item = grouped.setdefault(key, {"path": key, "flags": set(), "rows": []})
        item["flags"].update(parse_flags(row))
        item["rows"].append(row)

    items = list(grouped.values())
    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()

    def matching(item: dict[str, Any], term: str) -> bool:
        return any(term.lower() in flag.lower() for flag in item["flags"])

    for term, count in PRIORITY_TERMS:
        candidates = [item for item in items if matching(item, term) and item["path"] not in selected_paths]
        # For out-of-range and large boxes, prefer examples with the largest boxes.
        candidates.sort(
            key=lambda item: max(
                (area_of_box(parse_box(first_value(r, ("box_xywh_normalized", "box", "bbox", "bbox_xywh"))))
                 for r in item["rows"]),
                default=0.0,
            ),
            reverse=True,
        )
        limit = count if count else per_category
        if term == "VERY_LARGE_PEAR_BOX":
            limit = max(limit, min(4, per_category))
        for item in candidates[:limit]:
            item["priority_reason"] = term
            selected.append(item)
            selected_paths.add(item["path"])

    # If none of the expected flag names matched, still provide a few flagged examples.
    if not selected:
        for item in items[:max(1, per_category)]:
            item["priority_reason"] = "FLAGGED"
            selected.append(item)
    return selected


def find_image(path_text: str, csv_path: Path) -> Path | None:
    raw = Path(path_text)
    candidates = [raw]
    # If CSV paths are relative, try relative to the current folder and CSV folder.
    if not raw.is_absolute():
        candidates.extend([Path.cwd() / raw, csv_path.parent / raw, csv_path.parent.parent / raw])
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            pass
    return None


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def draw_marked_image(source: Image.Image, item: dict[str, Any]) -> Image.Image:
    image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(max(14, min(24, image.width // 45)))
    for row in item["rows"]:
        box = parse_box(first_value(row, ("box_xywh_normalized", "box", "bbox", "bbox_xywh")))
        if not box:
            continue
        cx, cy, bw, bh = box
        x1 = int(round((cx - bw / 2) * image.width))
        y1 = int(round((cy - bh / 2) * image.height))
        x2 = int(round((cx + bw / 2) * image.width))
        y2 = int(round((cy + bh / 2) * image.height))
        # Clamp only the drawing coordinates, not the CSV values.
        x1c, y1c = max(0, min(image.width - 1, x1)), max(0, min(image.height - 1, y1))
        x2c, y2c = max(0, min(image.width - 1, x2)), max(0, min(image.height - 1, y2))
        for offset in range(3):
            draw.rectangle((x1c - offset, y1c - offset, x2c + offset, y2c + offset), outline=(255, 40, 40))
        label = first_value(row, ("class_name", "class", "label", "class_name_text"))
        if not label:
            label = first_value(row, ("class_id", "class_id_int"))
        if label:
            draw.text((x1c + 3, max(0, y1c - 20)), label, fill=(255, 40, 40), font=font)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact contact sheet of suspicious dataset annotations.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Diagnostic CSV path (default: annotation_diagnostics/annotation_diagnostics.csv)")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Output folder (default: annotation_diagnostics/priority_review)")
    parser.add_argument("--per-category", type=int, default=6, help="Maximum examples per issue category")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.output)
    if not csv_path.is_file():
        print(f"ERROR: Cannot find diagnostic CSV:\n  {csv_path.resolve()}")
        print("Run this script from the backend folder, or use --csv with the full CSV path.")
        return 2

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print("ERROR: The CSV has no header row.")
                return 2
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        print(f"ERROR: Could not read CSV: {exc}")
        return 2

    if not rows:
        print("ERROR: The CSV contains no diagnostic rows.")
        return 2

    selected = select_rows(rows, max(1, args.per_category))
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "priority_review_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cards: list[tuple[Image.Image, str]] = []
    missing: list[str] = []
    summary_rows: list[dict[str, str]] = []

    for index, item in enumerate(selected, start=1):
        source_path = find_image(item["path"], csv_path)
        flag_text = ", ".join(sorted(item["flags"]))
        if source_path is None:
            missing.append(item["path"])
            summary_rows.append({"index": str(index), "image": item["path"], "flags": flag_text, "status": "IMAGE NOT FOUND"})
            continue
        try:
            with Image.open(source_path) as opened:
                marked = draw_marked_image(opened, item)
        except Exception as exc:
            missing.append(f"{item['path']} (could not open: {exc})")
            summary_rows.append({"index": str(index), "image": str(source_path), "flags": flag_text, "status": f"OPEN ERROR: {exc}"})
            continue

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.stem)[:80]
        target = images_dir / f"{index:02d}_{safe_name}.jpg"
        # Put a readable title strip on each individual image.
        title_h = 58
        canvas = Image.new("RGB", (marked.width, marked.height + title_h), "white")
        canvas.paste(marked, (0, title_h))
        d = ImageDraw.Draw(canvas)
        font = load_font(16)
        d.text((8, 5), f"{index:02d}. {source_path.name}", fill="black", font=font)
        d.text((8, 29), flag_text[:160], fill=(170, 20, 20), font=load_font(13))
        canvas.save(target, quality=92)
        cards.append((canvas.copy(), f"{index:02d}. {source_path.name}\n{flag_text}"))
        summary_rows.append({"index": str(index), "image": str(source_path), "flags": flag_text, "status": str(target)})

    if cards:
        thumb_w, thumb_h = 420, 300
        cols = 3
        rows_n = math.ceil(len(cards) / cols)
        sheet = Image.new("RGB", (cols * thumb_w, rows_n * thumb_h), (238, 238, 238))
        for i, (card, caption) in enumerate(cards):
            thumb = ImageOps.contain(card, (thumb_w - 12, thumb_h - 50))
            x = (i % cols) * thumb_w + (thumb_w - thumb.width) // 2
            y = (i // cols) * thumb_h + 5
            sheet.paste(thumb, (x, y))
            draw = ImageDraw.Draw(sheet)
            cap = caption.replace("\n", " | ")
            draw.text(((i % cols) * thumb_w + 8, (i // cols) * thumb_h + thumb_h - 42), cap[:110], fill=(20, 20, 20), font=load_font(12))
        sheet_path = out_dir / "priority_review_contact_sheet.jpg"
        sheet.save(sheet_path, quality=90)
    else:
        sheet_path = None

    summary_path = out_dir / "priority_review_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image", "flags", "status"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nPRIORITY REVIEW SET CREATED")
    print(f"Diagnostic rows read: {len(rows)}")
    print(f"Unique images selected: {len(selected)}")
    print(f"Images successfully prepared: {len(cards)}")
    if sheet_path:
        print(f"\nOPEN THIS FIRST:\n  {sheet_path.resolve()}")
    print(f"\nIndividual marked images:\n  {images_dir.resolve()}")
    print(f"Summary CSV:\n  {summary_path.resolve()}")
    if missing:
        print(f"\nImages that could not be loaded: {len(missing)}")
        for path in missing[:10]:
            print(f"  - {path}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    print("\nNo dataset labels or model weights were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
