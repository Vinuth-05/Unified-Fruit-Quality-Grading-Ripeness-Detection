from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Preview only. Does NOT change the dataset label.
image_path = Path(r"C:\Users\bdhul\Documents\Fruit_Grading_Project\Dataset_V3_Detection3Class_CLEAN\valid\images\image545_png.rf.9c29711cba2f9b9e312ab82b148996fe.jpg")
output_path = Path("pear_box_preview_tighter.jpg")

# Tighter approximate normalized YOLO xywh coordinates based on the visible image.
# Includes the pear and most of the visible stem. Verify visually before applying.
class_id = 2
cx, cy, bw, bh =  0.498047, 0.436198, 0.842448, 0.869792

if not image_path.is_file():
    raise SystemExit(f"Image not found: {image_path}")

im = Image.open(image_path).convert("RGB")
W, H = im.size
x1 = round((cx - bw / 2) * W)
y1 = round((cy - bh / 2) * H)
x2 = round((cx + bw / 2) * W)
y2 = round((cy + bh / 2) * H)

draw = ImageDraw.Draw(im)
draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=2)
try:
    font = ImageFont.truetype("arial.ttf", 10)
except OSError:
    font = ImageFont.load_default()
draw.rectangle((0, 0, W, 18), fill=(255, 255, 255))
draw.text((3, 3), f"Preview only | class {class_id} | {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}", fill=(180, 0, 0), font=font)
im.resize((768, 768)).save(output_path)
print(f"Preview saved: {output_path.resolve()}")
print("Proposed label (not applied):")
print(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
print("Inspect the red box before manually changing any label. Original annotation remains untouched.")
