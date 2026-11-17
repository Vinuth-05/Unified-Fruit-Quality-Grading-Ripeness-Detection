from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import shutil
import tempfile
from pathlib import Path

from inference import predict_image, predict_image_v3


# ============================================================
# FASTAPI APPLICATION
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Fruit Grading API",
    description=(
        "ViT-based fruit classification with YOLO multi-fruit "
        "detection, ripeness estimation and quality grading"
    ),
    version="3.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve generated annotated images at /outputs/<filename>
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# ============================================================
# HELPERS
# ============================================================

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def save_upload_to_temp(file: UploadFile) -> tuple[str, str]:
    """Save an uploaded image to a temporary file and return path/extension."""
    filename = file.filename or "uploaded_image"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Please upload JPG, JPEG, PNG or WEBP."
            ),
        )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    )
    temp_path = temp_file.name
    temp_file.close()

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return temp_path, filename


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Fruit Grading API is running",
        "model": "YOLO11n V3 + Vision Transformer V2",
        "version": app.version,
        "classes": 9,
        "detector_classes": ["Apple", "Orange", "Pear"],
        "endpoints": {
            "health": "/health",
            "single_fruit_prediction": "/predict",
            "multi_fruit_prediction": "/predict-v3",
            "interactive_docs": "/docs",
        },
        "features": [
            "Fruit detection",
            "Ripeness classification",
            "Project-defined quality grading",
            "Separate detector and classifier confidence",
            "Multi-fruit counting",
            "Manual review warnings for model disagreements",
            "Annotated image output",
        ],
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "fruit-grading-api",
    }


# ============================================================
# LEGACY SINGLE-IMAGE VIT PREDICTION
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None

    try:
        temp_path, original_filename = save_upload_to_temp(file)
        result = predict_image(temp_path)

        return {
            "success": True,
            "filename": original_filename,
            "prediction": result,
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Single-image prediction failed: {error}",
        ) from error
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()


# ============================================================
# V3 MULTI-FRUIT PREDICTION
# ============================================================

@app.post("/predict-v3")
async def predict_v3(file: UploadFile = File(...)):
    temp_path = None

    try:
        temp_path, original_filename = save_upload_to_temp(file)

        # The inference function writes the annotated image beside the
        # temporary upload. Copy it to a persistent backend output folder
        # before the temporary upload is removed.
        result = predict_image_v3(
            temp_path,
            output_dir=str(OUTPUT_DIR),
        )

        annotated_path = result.get("annotated_image_path")
        annotated_filename = (
            Path(annotated_path).name if annotated_path else None
        )

        return {
            "success": True,
            "filename": original_filename,
            "prediction": result,
            "annotated_image_url": (
                f"/outputs/{annotated_filename}"
                if annotated_filename else None
            ),
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"V3 multi-fruit prediction failed: {error}",
        ) from error
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()
