from ultralytics import YOLO
from pathlib import Path
DATA = r"C:\Users\bdhul\Documents\Fruit_Grading_Project\Dataset_V3_Detection3Class_CLEAN\data.yaml"
WEIGHTS = r"C:\Users\bdhul\Documents\Fruit_Grading_Project\Review01_App\backend\models\best.pt"
OUTPUT = r"C:\Users\bdhul\Documents\Fruit_Grading_Project\Review01_App\backend\runs\pear_finetune"
model = YOLO(WEIGHTS)
model.train(
    data=DATA,
    epochs=15,
    imgsz=640,
    batch=8,
    patience=5,
    workers=2,
    seed=42,
    project=OUTPUT,
    name="candidate",
    exist_ok=False,
    plots=True,
    save=True
)
