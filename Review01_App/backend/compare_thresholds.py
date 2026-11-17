
from ultralytics import YOLO
from pathlib import Path

model = YOLO("models/best.pt")

image_folder = Path(r"C:\Users\bdhul\OneDrive\Pictures")
extensions = {".jpg", ".jpeg", ".png", ".webp"}

images = [
    p for p in image_folder.iterdir()
    if p.suffix.lower() in extensions
][:10]

for image_path in images:
    result = model.predict(
        source=str(image_path),
        conf=0.01,
        iou=0.45,
        verbose=False
    )[0]

    print(f"\nImage: {image_path.name}")

    if len(result.boxes) == 0:
        print("No detections even at 0.01 confidence.")
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names[class_id]

        print(f"{label}: {confidence:.2%}")

print("\nDiagnostic complete.")
