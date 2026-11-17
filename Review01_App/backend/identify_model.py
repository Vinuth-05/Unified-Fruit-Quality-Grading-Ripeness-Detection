
from ultralytics import YOLO
from pathlib import Path

model_path = Path("models/best.pt")

print("=" * 50)
print("MODEL IDENTIFICATION")
print("=" * 50)

if not model_path.exists():
    print("Model file not found:", model_path.resolve())
    raise SystemExit

print("File:", model_path.resolve())
print(f"Size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

try:
    model = YOLO(str(model_path))

    print("\nRESULT: This file loads as a YOLO model.")
    print("Model type:", model.task)
    print("Class names:", model.names)

except Exception as e:
    print("\nThis file did not load as a YOLO model.")
    print("Error:", str(e)[:500])
    print("It may be a PyTorch checkpoint, but its architecture")
    print("must be checked separately.")
