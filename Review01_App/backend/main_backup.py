from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import os
import shutil
import tempfile

from inference import predict_image


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Fruit Grading API",
    description=(
        "ViT-based unified fruit classification, "
        "ripeness detection and quality grading API"
    ),
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "Fruit Grading API is running",

        "model":
            "Vision Transformer (ViT)",

        "classes":
            9,

        "features": [

            "Fruit classification",

            "Ripeness detection",

            "Quality grading",

            "Confidence analysis",

            "Prediction reliability"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "service":
            "fruit-grading-api"
    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # ALLOWED IMAGE FORMATS
    # --------------------------------------------------------

    allowed_extensions = [

        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]


    file_extension = os.path.splitext(
        file.filename
    )[1].lower()


    if file_extension not in allowed_extensions:

        return {

            "success":
                False,

            "error":
                "Unsupported image format. "
                "Please upload JPG, JPEG, PNG or WEBP."
        }


    # --------------------------------------------------------
    # TEMPORARY FILE
    # --------------------------------------------------------

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    )

    temp_path = temp_file.name

    temp_file.close()


    try:

        # ----------------------------------------------------
        # SAVE UPLOADED IMAGE
        # ----------------------------------------------------

        with open(
            temp_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        result = predict_image(
            temp_path
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "success":
                True,

            "filename":
                file.filename,

            "prediction":
                result
        }


    except Exception as error:

        return {

            "success":
                False,

            "error":
                str(error)
        }


    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY IMAGE
        # ----------------------------------------------------

        if os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )