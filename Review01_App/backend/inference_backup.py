import os

import torch

from PIL import Image

from transformers import (
    AutoImageProcessor,
    ViTForImageClassification
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_PATH = r"C:\Users\bdhul\Documents\Fruit_Grading_Project"

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "Best_ViT_Model_V2"
)


# ============================================================
# OPTIONAL VALIDATION DATASET
# ============================================================
#
# If your validation dataset is somewhere else, put its path
# here.
#
# Expected structure:
#
# validation/
# ├── Apple_Overripe/
# ├── Apple_Ripe/
# ├── Apple_Unripe/
# ├── Orange_Overripe/
# ├── Orange_Ripe/
# ├── Orange_Unripe/
# ├── Pear_Overripe/
# ├── Pear_Ripe/
# └── Pear_Unripe/
#
# Leave as None if you don't want to use calibration yet.
#
# ============================================================

VALIDATION_DATASET_PATH = None


# ============================================================
# CLASS CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Apple_Overripe",
    "Apple_Ripe",
    "Apple_Unripe",
    "Orange_Overripe",
    "Orange_Ripe",
    "Orange_Unripe",
    "Pear_Overripe",
    "Pear_Ripe",
    "Pear_Unripe"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ViT model...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_PATH
)

model = ViTForImageClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("ViT model loaded successfully.")
print("Device:", DEVICE)
print("Classes:", len(CLASS_NAMES))


# ============================================================
# QUALITY GRADING SYSTEM
# ============================================================

GRADING_RULES = {

    "Ripe": {
        "grade": "Grade A",
        "assessment": "Optimal ripeness",
        "recommendation":
            "Ready for consumption or sale"
    },

    "Unripe": {
        "grade": "Grade B",
        "assessment":
            "Requires further ripening",
        "recommendation":
            "Allow further ripening before consumption"
    },

    "Overripe": {
        "grade": "Grade C",
        "assessment":
            "Beyond optimal ripeness",
        "recommendation":
            "Process immediately or use for value-added products"
    }
}


# ============================================================
# PHASE 2 REJECTION CONFIGURATION
# ============================================================

# These are deliberately kept separate from the old
# "prediction status" thresholds.
#
# The final values can be calibrated using your validation
# dataset after we inspect the real distributions.

MIN_CONFIDENCE = 70.0
MIN_MARGIN = 10.0


# ============================================================
# CONFIDENCE STATUS
# ============================================================

def get_prediction_status(confidence):

    if confidence >= 80:
        return "Reliable prediction"

    elif confidence >= 60:
        return "Moderate confidence"

    else:
        return "Low confidence"


# ============================================================
# UNKNOWN / REJECTION DECISION
# ============================================================

def determine_rejection(
    confidence,
    margin
):
    """
    Phase 2 rejection logic.

    confidence:
        Probability of the top predicted class.

    margin:
        Difference between top-1 and top-2 probabilities.

    Both signals are considered.

    This prevents the system from relying only on the
    top-class softmax confidence.
    """

    if confidence < MIN_CONFIDENCE:

        return True, "Low prediction confidence"

    if margin < MIN_MARGIN:

        return True, "Low separation between competing classes"

    return False, "Prediction passed rejection check"


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path: str):

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )


    # --------------------------------------------------------
    # SORT PREDICTIONS
    # --------------------------------------------------------

    sorted_values, sorted_indices = torch.sort(
        probabilities[0],
        descending=True
    )


    # --------------------------------------------------------
    # TOP-1
    # --------------------------------------------------------

    predicted_index = (
        sorted_indices[0].item()
    )

    confidence = (
        sorted_values[0].item() * 100
    )


    # --------------------------------------------------------
    # TOP-2
    # --------------------------------------------------------

    second_confidence = (
        sorted_values[1].item() * 100
    )


    # --------------------------------------------------------
    # TOP-1 / TOP-2 MARGIN
    # --------------------------------------------------------

    margin = (
        confidence -
        second_confidence
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    # --------------------------------------------------------
    # UNKNOWN / REJECTION CHECK
    # --------------------------------------------------------

    is_rejected, rejection_reason = (
        determine_rejection(
            confidence,
            margin
        )
    )


    # --------------------------------------------------------
    # TOP 3 CONFIDENCE DISTRIBUTION
    # --------------------------------------------------------

    confidence_distribution = []

    for rank in range(3):

        index = sorted_indices[
            rank
        ].item()

        value = sorted_values[
            rank
        ].item() * 100

        confidence_distribution.append(
            {
                "class_name":
                    CLASS_NAMES[index],

                "confidence":
                    round(value, 2)
            }
        )


    # ========================================================
    # UNKNOWN / REJECTED RESULT
    # ========================================================

    if is_rejected:

        return {

            "fruit_type":
                "Unknown",

            "ripeness":
                "Unknown",

            "confidence":
                round(confidence, 2),

            "class_name":
                predicted_class,

            "quality_grade":
                "Unknown",

            "quality_assessment":
                "Unable to reliably identify a supported fruit",

            "recommendation":
                "Manual inspection required",

            "prediction_status":
                "Unknown / Rejected",

            "is_unknown":
                True,

            "rejection_reason":
                rejection_reason,

            "top_2_confidence":
                round(second_confidence, 2),

            "prediction_margin":
                round(margin, 2),

            "confidence_distribution":
                confidence_distribution
        }


    # ========================================================
    # KNOWN / ACCEPTED RESULT
    # ========================================================

    fruit_type, ripeness = (
        predicted_class.split(
            "_",
            1
        )
    )


    # --------------------------------------------------------
    # QUALITY GRADING
    # --------------------------------------------------------

    grading = GRADING_RULES.get(
        ripeness,

        {
            "grade":
                "Unknown",

            "assessment":
                "Unable to determine quality",

            "recommendation":
                "Manual inspection required"
        }
    )


    # --------------------------------------------------------
    # CONFIDENCE STATUS
    # --------------------------------------------------------

    prediction_status = (
        get_prediction_status(
            confidence
        )
    )


    # --------------------------------------------------------
    # FINAL KNOWN RESULT
    # --------------------------------------------------------

    return {

        "fruit_type":
            fruit_type,

        "ripeness":
            ripeness,

        "confidence":
            round(confidence, 2),

        "class_name":
            predicted_class,

        "quality_grade":
            grading["grade"],

        "quality_assessment":
            grading["assessment"],

        "recommendation":
            grading["recommendation"],

        "prediction_status":
            prediction_status,

        "is_unknown":
            False,

        "rejection_reason":
            None,

        "top_2_confidence":
            round(second_confidence, 2),

        "prediction_margin":
            round(margin, 2),

        "confidence_distribution":
            confidence_distribution
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "Enter image path: "
    ).strip().strip('"')


    try:

        result = predict_image(
            image_path
        )


        print("\n")

        print("=" * 70)

        print(
            "FRUIT QUALITY GRADING RESULT"
        )

        print("=" * 70)


        print(
            "Fruit Type          :",
            result["fruit_type"]
        )

        print(
            "Ripeness            :",
            result["ripeness"]
        )

        print(
            "Confidence          :",
            f'{result["confidence"]:.2f}%'
        )

        print(
            "Predicted Class     :",
            result["class_name"]
        )

        print(
            "Quality Grade       :",
            result["quality_grade"]
        )

        print(
            "Assessment          :",
            result["quality_assessment"]
        )

        print(
            "Recommendation      :",
            result["recommendation"]
        )

        print(
            "Prediction Status   :",
            result["prediction_status"]
        )

        print(
            "Unknown / Rejected  :",
            result["is_unknown"]
        )

        print(
            "Top-2 Confidence    :",
            f'{result["top_2_confidence"]:.2f}%'
        )

        print(
            "Prediction Margin   :",
            f'{result["prediction_margin"]:.2f}%'
        )


        if result["rejection_reason"]:

            print(
                "Rejection Reason    :",
                result["rejection_reason"]
            )


        print("\n")

        print(
            "Confidence Distribution"
        )

        print("-" * 70)


        for item in result[
            "confidence_distribution"
        ]:

            print(
                f'{item["class_name"]:<25}'
                f'{item["confidence"]:.2f}%'
            )


        print("=" * 70)


    except Exception as error:

        print(
            "\nERROR:",
            error
        )