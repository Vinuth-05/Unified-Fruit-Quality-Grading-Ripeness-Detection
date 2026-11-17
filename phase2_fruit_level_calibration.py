import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

PROJECT_PATH = r"C:\Users\bdhul\Documents\Fruit_Grading_Project"

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "Best_ViT_Model_V2"
)

VALID_PATH = os.path.join(
    PROJECT_PATH,
    "Prepared_Dataset",
    "valid"
)

UNKNOWN_PATH = r"C:\Users\bdhul\OneDrive\REVIEW_02_IMAGES"


# ============================================================
# MODEL / CLASS CONFIGURATION
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

FRUIT_NAMES = [
    "Apple",
    "Orange",
    "Pear"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("PHASE 2 - FRUIT LEVEL FEATURE CALIBRATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_PATH
)

print("Loading ViT model...")

model = ViTForImageClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_feature(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():

        outputs = model.vit(
            pixel_values=pixel_values
        )

        # CLS token
        feature = outputs.last_hidden_state[:, 0, :]

        # L2 normalization
        feature = torch.nn.functional.normalize(
            feature,
            p=2,
            dim=1
        )

    return feature[0].cpu().numpy()


# ============================================================
# PREDICTION
# ============================================================

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    predicted_index = torch.argmax(probabilities).item()

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = probabilities[predicted_index].item() * 100

    return predicted_class, confidence


# ============================================================
# FRUIT EXTRACTION
# ============================================================

def get_fruit_from_class(class_name):

    return class_name.split("_")[0]


# ============================================================
# COSINE DISTANCE
# ============================================================

def cosine_distance(a, b):

    similarity = np.dot(a, b)

    return 1.0 - similarity


# ============================================================
# BUILD FRUIT-LEVEL PROFILES
# ============================================================

print("=" * 70)
print("BUILDING FRUIT-LEVEL FEATURE PROFILES")
print("=" * 70)

fruit_features = defaultdict(list)

validation_counts = defaultdict(int)

for class_name in CLASS_NAMES:

    class_path = os.path.join(
        VALID_PATH,
        class_name
    )

    if not os.path.exists(class_path):

        print(
            f"WARNING: Missing folder: {class_path}"
        )

        continue

    files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    print(
        f"{class_name:<20} : {len(files)} images"
    )

    fruit_name = get_fruit_from_class(
        class_name
    )

    for filename in files:

        image_path = os.path.join(
            class_path,
            filename
        )

        try:

            feature = extract_feature(
                image_path
            )

            fruit_features[fruit_name].append(
                feature
            )

            validation_counts[fruit_name] += 1

        except Exception as e:

            print(
                f"ERROR processing {image_path}: {e}"
            )


print()


# ============================================================
# CREATE CENTROIDS
# ============================================================

fruit_centroids = {}

print("=" * 70)
print("FRUIT-LEVEL CENTROIDS")
print("=" * 70)

for fruit_name in FRUIT_NAMES:

    features = np.array(
        fruit_features[fruit_name]
    )

    centroid = np.mean(
        features,
        axis=0
    )

    # Normalize centroid
    centroid = centroid / (
        np.linalg.norm(centroid) + 1e-12
    )

    fruit_centroids[fruit_name] = centroid

    print(
        f"{fruit_name:<10} : "
        f"{len(features)} validation images"
    )

print()


# ============================================================
# CALIBRATION USING CORRECT VALIDATION PREDICTIONS
# ============================================================

print("=" * 70)
print("CALIBRATING FRUIT-LEVEL DISTANCE")
print("=" * 70)

correct_distances = []

total_validation = 0
correct_validation = 0

for class_name in CLASS_NAMES:

    class_path = os.path.join(
        VALID_PATH,
        class_name
    )

    if not os.path.exists(class_path):
        continue

    true_fruit = get_fruit_from_class(
        class_name
    )

    files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    for filename in files:

        image_path = os.path.join(
            class_path,
            filename
        )

        try:

            predicted_class, confidence = predict(
                image_path
            )

            predicted_fruit = get_fruit_from_class(
                predicted_class
            )

            total_validation += 1

            # Only use correctly classified
            # validation samples for threshold calibration.
            if predicted_fruit == true_fruit:

                correct_validation += 1

                feature = extract_feature(
                    image_path
                )

                distances = {}

                for fruit_name in FRUIT_NAMES:

                    distances[fruit_name] = cosine_distance(
                        feature,
                        fruit_centroids[fruit_name]
                    )

                nearest_fruit = min(
                    distances,
                    key=distances.get
                )

                nearest_distance = distances[
                    nearest_fruit
                ]

                # Only consider the sample if
                # the nearest fruit is actually correct.
                if nearest_fruit == true_fruit:

                    correct_distances.append(
                        nearest_distance
                    )

        except Exception as e:

            print(
                f"ERROR during calibration: {e}"
            )


correct_distances = np.array(
    correct_distances
)

print(
    f"Total validation images : {total_validation}"
)

print(
    f"Correct fruit predictions: {correct_validation}"
)

print(
    f"Samples used for calibration: "
    f"{len(correct_distances)}"
)

if len(correct_distances) == 0:

    raise RuntimeError(
        "No correct validation samples available "
        "for calibration."
    )


print()

print(
    f"Minimum distance : "
    f"{np.min(correct_distances):.4f}"
)

print(
    f"Maximum distance : "
    f"{np.max(correct_distances):.4f}"
)

print(
    f"Mean distance    : "
    f"{np.mean(correct_distances):.4f}"
)

print(
    f"95th percentile  : "
    f"{np.percentile(correct_distances, 95):.4f}"
)

print(
    f"99th percentile  : "
    f"{np.percentile(correct_distances, 99):.4f}"
)


# ============================================================
# THRESHOLD
# ============================================================

THRESHOLD = np.percentile(
    correct_distances,
    99
)

print()

print(
    f"FRUIT-LEVEL REJECTION THRESHOLD: "
    f"{THRESHOLD:.4f}"
)

print(
    "Threshold source: 99th percentile "
    "of correctly classified validation samples."
)

print()


# ============================================================
# TEST UNKNOWN / OUT-OF-DOMAIN IMAGES
# ============================================================

print("=" * 70)
print("TESTING PHASE 2 UNKNOWN / OOD IMAGES")
print("=" * 70)

if not os.path.exists(UNKNOWN_PATH):

    raise FileNotFoundError(
        f"Unknown image folder not found:\n{UNKNOWN_PATH}"
    )


files = [
    f for f in os.listdir(UNKNOWN_PATH)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
]

files.sort()


accepted = 0
rejected = 0


for filename in files:

    image_path = os.path.join(
        UNKNOWN_PATH,
        filename
    )

    try:

        predicted_class, confidence = predict(
            image_path
        )

        predicted_fruit = get_fruit_from_class(
            predicted_class
        )

        feature = extract_feature(
            image_path
        )

        distances = {}

        for fruit_name in FRUIT_NAMES:

            distances[fruit_name] = cosine_distance(
                feature,
                fruit_centroids[fruit_name]
            )

        nearest_fruit = min(
            distances,
            key=distances.get
        )

        nearest_distance = distances[
            nearest_fruit
        ]

        is_rejected = (
            nearest_distance > THRESHOLD
        )

        if is_rejected:
            status = "UNKNOWN / REJECTED"
            rejected += 1
        else:
            status = "KNOWN / ACCEPTED"
            accepted += 1

        print("-" * 70)

        print(
            f"Image            : {filename}"
        )

        print(
            f"ViT Prediction   : {predicted_class}"
        )

        print(
            f"ViT Confidence   : {confidence:.2f}%"
        )

        print(
            f"Predicted Fruit  : {predicted_fruit}"
        )

        print(
            f"Nearest Fruit    : {nearest_fruit}"
        )

        print(
            f"Fruit Distance   : {nearest_distance:.4f}"
        )

        print(
            f"Threshold        : {THRESHOLD:.4f}"
        )

        print(
            f"FINAL STATUS     : {status}"
        )

    except Exception as e:

        print("-" * 70)

        print(
            f"ERROR processing {filename}: {e}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FRUIT-LEVEL OOD SUMMARY")
print("=" * 70)

print(
    f"Total unknown/OOD images : {len(files)}"
)

print(
    f"Accepted                 : {accepted}"
)

print(
    f"Rejected                 : {rejected}"
)

if len(files) > 0:

    rejection_rate = (
        rejected / len(files)
    ) * 100

    print(
        f"Rejection rate           : "
        f"{rejection_rate:.2f}%"
    )

print()

print("=" * 70)
print("CALIBRATION COMPLETE")
print("=" * 70)

print(
    "Phase 1 model was NOT modified."
)

print(
    "Training dataset was NOT modified."
)

print(
    "Only the validation set was used "
    "for calibration."
)

print("=" * 70)