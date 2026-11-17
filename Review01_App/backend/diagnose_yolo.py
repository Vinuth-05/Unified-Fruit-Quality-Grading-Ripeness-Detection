"""
YOLO V3 RAW DETECTION DIAGNOSTIC
--------------------------------
Run this script against an image to inspect YOLO predictions BEFORE ViT
classification. It does not modify or overwrite either trained model.

Usage (from your backend folder):
    python diagnose_yolo.py "C:\path\to\apple_only_image.jpg"

Optional:
    python diagnose_yolo.py "C:\path\to\image.jpg" --conf 0.40 --iou 0.45

It saves:
  - <image>_yolo_diagnostic.jpg : YOLO-only boxes and class/confidence labels
  - <image>_yolo_diagnostic.txt : readable list of raw detections
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def get_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description="Inspect raw YOLO fruit detections.")
    parser.add_argument("image", help="Path to the test image")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "models" / "best.pt"),
                        help="Path to YOLO best.pt")
    args = parser.parse_args()

    image_path = Path(args.image.strip('"'))
    model_path = Path(args.model.strip('"'))

    if not image_path.is_file():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {model_path}\n"
            "Expected backend\\models\\best.pt, or pass --model PATH"
        )
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1")

    model = YOLO(str(model_path))
    with Image.open(image_path) as src:
        image = src.convert("RGB")

    results = model.predict(
        source=image,
        conf=args.conf,
        iou=args.iou,
        verbose=False
    )
    result = results[0]
    names = model.names

    records = []
    if result.boxes is not None:
        for idx, box in enumerate(result.boxes, start=1):
            xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item()) * 100
            class_name = names.get(class_id, str(class_id))
            records.append((idx, class_name, confidence, xyxy))

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = get_font(16)
    palette = {
        "Apple": (40, 190, 70),
        "Orange": (245, 140, 25),
        "Pear": (50, 150, 235),
    }

    for idx, class_name, confidence, coords in records:
        x1, y1, x2, y2 = [int(round(v)) for v in coords]
        color = palette.get(class_name, (220, 60, 60))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{class_name} | YOLO {confidence:.1f}%"
        try:
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)
        label_top = max(0, y1 - text_h - 6)
        label_right = min(annotated.width, x1 + text_w + 8)
        draw.rectangle(
            (x1, label_top, label_right, label_top + text_h + 6),
            fill=color
        )
        draw.text((x1 + 4, label_top + 2), label, fill=(15, 15, 15), font=font)

    output_image = image_path.with_name(f"{image_path.stem}_yolo_diagnostic.jpg")
    output_text = image_path.with_name(f"{image_path.stem}_yolo_diagnostic.txt")
    annotated.save(output_image, quality=94)

    lines = [
        "YOLO RAW DETECTION DIAGNOSTIC",
        f"Image: {image_path}",
        f"Model: {model_path}",
        f"Confidence threshold: {args.conf:.2f}",
        f"NMS IoU threshold: {args.iou:.2f}",
        f"Total detections: {len(records)}",
        f"Class names: {names}",
        "",
    ]
    for idx, class_name, confidence, coords in records:
        lines.append(
            f"{idx}. class={class_name}, confidence={confidence:.2f}%, "
            f"box=[{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]"
        )
    if not records:
        lines.append("No detections passed the selected confidence threshold.")

    output_text.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nAnnotated image saved to: {output_image}")
    print(f"Detection report saved to: {output_text}")


if __name__ == "__main__":
    main()
