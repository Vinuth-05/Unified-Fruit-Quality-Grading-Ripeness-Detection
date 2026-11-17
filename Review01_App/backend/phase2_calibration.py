import os
import numpy as np
import torch

from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification


# ============================================================
# PHASE 2 CALIBRATION CONFIGURATION
# ============================================================

PROJECT_PATH = r"C:\Users\bdhul\Documents\Fruit_Grading_Project"

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "Best_ViT_Model_V2"
)

VALIDATION_PATH = os.path.join(
    PROJECT_PATH,
    "Prepared_Dataset",
    "valid"
)

UNKNOWN_PATH = r"C:\Users\bdhul\OneDrive\REVIEW_02_IMAGES"


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

print("=" * 75)
print("PHASE 2 UNKNOWN DETECTION CALIBRATION")
print("=" * 75)

print("\nLoading ViT model...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_PATH
)

model = ViTForImageClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_feature(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.vit(
            **inputs
        )

    # CLS token representation
    feature = outputs.last_hidden_state[:, 0, :]

    feature = torch.nn.functional.normalize(
        feature,
        p=2,
        dim=1
    )

    return feature[0].cpu().numpy()


# ============================================================
# COSINE DISTANCE
# ============================================================

def cosine_distance(a, b):

    similarity = np.dot(a, b)

    return 1.0 - similarity


# ============================================================
# COLLECT VALIDATION FEATURES
# ============================================================

print("\n")
print("=" * 75)
print("STEP 1: EXTRACTING VALIDATION FEATURES")
print("=" * 75)

class_features = {
    class_name: []
    for class_name in CLASS_NAMES
}


total_images = 0


for class_name in CLASS_NAMES:

    class_path = os.path.join(
        VALIDATION_PATH,
        class_name
    )

    if not os.path.isdir(class_path):

        print(
            f"WARNING: Missing class folder: {class_path}"
        )

        continue


    image_files = [

        filename

        for filename in os.listdir(
            class_path
        )

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            )
        )
    ]


    print(
        f"{class_name:<25} : "
        f"{len(image_files)} images"
    )


    for filename in image_files:

        image_path = os.path.join(
            class_path,
            filename
        )

        try:

            feature = extract_feature(
                image_path
            )

            class_features[
                class_name
            ].append(feature)

            total_images += 1

        except Exception as error:

            print(
                "ERROR:",
                image_path,
                error
            )


print("\nTotal validation images:", total_images)


# ============================================================
# BUILD CLASS CENTROIDS
# ============================================================

print("\n")
print("=" * 75)
print("STEP 2: BUILDING KNOWN-CLASS FEATURE PROFILES")
print("=" * 75)


class_centroids = {}


for class_name in CLASS_NAMES:

    features = class_features[
        class_name
    ]

    if len(features) == 0:

        print(
            f"WARNING: No features for {class_name}"
        )

        continue


    features_array = np.vstack(
        features
    )

    centroid = np.mean(
        features_array,
        axis=0
    )

    # Normalize centroid
    norm = np.linalg.norm(
        centroid
    )

    if norm > 0:

        centroid = (
            centroid / norm
        )

    class_centroids[
        class_name
    ] = centroid


    print(
        f"{class_name:<25} : "
        f"{len(features)} features"
    )


# ============================================================
# VALIDATION DISTANCE ANALYSIS
# ============================================================

print("\n")
print("=" * 75)
print("STEP 3: VALIDATION DISTANCE ANALYSIS")
print("=" * 75)


correct_distances = []
wrong_distances = []


for actual_class in CLASS_NAMES:

    for feature in class_features.get(
        actual_class,
        []
    ):

        distances = {}

        for class_name, centroid in (
            class_centroids.items()
        ):

            distances[class_name] = (
                cosine_distance(
                    feature,
                    centroid
                )
            )


        nearest_class = min(
            distances,
            key=distances.get
        )

        nearest_distance = (
            distances[
                nearest_class
            ]
        )


        if nearest_class == actual_class:

            correct_distances.append(
                nearest_distance
            )

        else:

            wrong_distances.append(
                nearest_distance
            )


# ============================================================
# PRINT DISTANCE STATISTICS
# ============================================================

print("\nCorrect validation predictions:")
print(
    "Count:",
    len(correct_distances)
)


if correct_distances:

    print(
        "Minimum distance :",
        f"{min(correct_distances):.4f}"
    )

    print(
        "Maximum distance :",
        f"{max(correct_distances):.4f}"
    )

    print(
        "Mean distance    :",
        f"{np.mean(correct_distances):.4f}"
    )

    print(
        "95th percentile  :",
        f"{np.percentile(correct_distances, 95):.4f}"
    )

    print(
        "99th percentile  :",
        f"{np.percentile(correct_distances, 99):.4f}"
    )


# ============================================================
# DATA-DERIVED REJECTION BOUNDARY
# ============================================================

if correct_distances:

    rejection_threshold = float(
        np.percentile(
            correct_distances,
            99
        )
    )

else:

    rejection_threshold = 0.30


print("\n")
print("=" * 75)
print("PHASE 2 FEATURE REJECTION THRESHOLD")
print("=" * 75)

print(
    "Threshold:",
    f"{rejection_threshold:.4f}"
)

print(
    "Derived from:",
    "99th percentile of correctly classified validation distances"
)


# ============================================================
# UNKNOWN IMAGE TESTING
# ============================================================

print("\n")
print("=" * 75)
print("STEP 4: TESTING REVIEW_02 UNKNOWN IMAGES")
print("=" * 75)


unknown_files = [

    filename

    for filename in os.listdir(
        UNKNOWN_PATH
    )

    if filename.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        )
    )
]


if len(unknown_files) == 0:

    print(
        "No image files found in:"
    )

    print(
        UNKNOWN_PATH
    )


for filename in unknown_files:

    image_path = os.path.join(
        UNKNOWN_PATH,
        filename
    )


    try:

        # ----------------------------------------------------
        # FEATURE
        # ----------------------------------------------------

        feature = extract_feature(
            image_path
        )


        # ----------------------------------------------------
        # DISTANCES
        # ----------------------------------------------------

        distances = {}

        for class_name, centroid in (
            class_centroids.items()
        ):

            distances[class_name] = (
                cosine_distance(
                    feature,
                    centroid
                )
            )


        nearest_class = min(
            distances,
            key=distances.get
        )

        nearest_distance = (
            distances[
                nearest_class
            ]
        )


        # ----------------------------------------------------
        # MODEL CLASSIFICATION
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        inputs = processor(
            images=image,
            return_tensors="pt"
        )


        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }


        with torch.no_grad():

            outputs = model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )


        predicted_index = torch.argmax(
            probabilities,
            dim=-1
        ).item()


        predicted_class = CLASS_NAMES[
            predicted_index
        ]


        confidence = (
            probabilities[
                0,
                predicted_index
            ].item() * 100
        )


        # ----------------------------------------------------
        # REJECTION DECISION
        # ----------------------------------------------------

        if nearest_distance > rejection_threshold:

            status = "UNKNOWN / REJECTED"

        else:

            status = "KNOWN / ACCEPTED"


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        print("\n")
        print("-" * 75)

        print(
            "Image:",
            filename
        )

        print(
            "ViT Prediction:",
            predicted_class
        )

        print(
            "Softmax Confidence:",
            f"{confidence:.2f}%"
        )

        print(
            "Nearest Known Class:",
            nearest_class
        )

        print(
            "Feature Distance:",
            f"{nearest_distance:.4f}"
        )

        print(
            "Rejection Threshold:",
            f"{rejection_threshold:.4f}"
        )

        print(
            "FINAL STATUS:",
            status
        )

        print("-" * 75)


    except Exception as error:

        print("\nERROR:")
        print(image_path)
        print(error)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 75)
print("PHASE 2 CALIBRATION COMPLETE")
print("=" * 75)

print(
    "The original ViT model was NOT modified."
)

print(
    "Training and test datasets were NOT modified."
)

print(
    "Only the validation set was used for calibration."
)

print("=" * 75)