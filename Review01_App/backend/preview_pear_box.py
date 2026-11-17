from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Run this from your backend folder. This only creates a preview image.
image_path = Path(r"C:\Users\bdhul\Documents\Fruit_Grading_Project\Dataset_V3_Detection3Class_CLEAN\valid\images\image545_png.rf.9c29711cba2f9b9e312ab82b148996fe.jpg")
output_path = Path("pear_box_preview.jpg")

# Approximate normalized YOLO xywh box estimated from the image.
# It includes the visible pear and its stem; inspect the preview before changing labels.
class_id = 2
cx, cy, bw, bh = 0.493000, 0.434000, 0.870000, 0.870000

if not image_path.is_file():
    raise SystemExit(f"Image not found: {image_path}")

im = Image.open(image_path).convert("RGB")
W, H = im.size
x1 = round((cx - bw / 2) * W)
y1 = round((cy - bh / 2) * H)
x2 = round((cx + bw / 2) * W)
y2 = round((cy + bh / 2) * H)

draw = ImageDraw.Draw(im)
# Draw unclipped intended bounds in orange, and clipped visible bounds in red.
draw.rectangle((x1, y1, x2, y2), outline=(255, 165, 0), width=3)
draw.rectangle((max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)), outline=(255, 40, 40), width=3)
try:
    font = ImageFont.truetype("arial.ttf", 12)
except OSError:
    font = ImageFont.load_default()
draw.text((5, 5), f"Suggested Pear box | class {class_id} | xywh={cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}", fill=(255, 40, 40), font=font)
im.resize((768, 768)).save(output_path)
print(f"Preview saved: {output_path.resolve()}")
print("Suggested label (NOT applied):")
print(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
print("Inspect the preview before editing the original label. This script does not modify labels.")
