import os
import torch
import numpy as np

from PIL import Image
from torchvision import models, transforms
from torch import nn


# ============================================================
# PATHS
# ============================================================

PROJECT_PATH = r"C:\Users\bdhul\Documents\Fruit_Grading_Project"

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "Fruit_Verifier_Model",
    "fruit_verifier_resnet18.pth"
)

VALID_PATH = os.path.join(
    PROJECT_PATH,
    "Prepared_Dataset",
    "valid"
)

OOD_PATH = r"C:\Users\bdhul\OneDrive\REVIEW_02_IMAGES"


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Apple",
    "Orange",
    "Pear"
]

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("PHASE 2 - FRUIT VERIFIER THRESHOLD CALIBRATION")
print("=" * 80)

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_PATH}")
print(f"Valid  : {VALID_PATH}")
print(f"OOD    : {OOD_PATH}")
print()


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Verifier model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(VALID_PATH):
    raise FileNotFoundError(
        f"Validation folder not found:\n{VALID_PATH}"
    )

if not os.path.exists(OOD_PATH):
    raise FileNotFoundError(
        f"OOD folder not found:\n{OOD_PATH}"
    )


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading fruit verifier...")

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Verifier loaded successfully.")

print(
    f"Best validation accuracy: "
    f"{checkpoint.get('best_val_accuracy', 'N/A'):.2f}%"
)

print()


# ============================================================
# PREDICTION
# ============================================================

def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        probabilities[
            predicted_index
        ].item() * 100
    )

    return predicted_class, confidence


# ============================================================
# KNOWN VALIDATION SET
# ============================================================

print("=" * 80)
print("CALIBRATING USING KNOWN VALIDATION IMAGES")
print("=" * 80)

known_correct_confidences = []

known_incorrect_confidences = []

total_known = 0
correct_known = 0
incorrect_known = 0


for fruit_name in CLASS_NAMES:

    # The original dataset has three ripeness folders
    # for every fruit.
    fruit_prefix = fruit_name + "_"

    matching_folders = [

        folder

        for folder in os.listdir(
            VALID_PATH
        )

        if folder.startswith(
            fruit_prefix
        )
        and os.path.isdir(
            os.path.join(
                VALID_PATH,
                folder
            )
        )
    ]

    fruit_total = 0
    fruit_correct = 0

    for folder in matching_folders:

        folder_path = os.path.join(
            VALID_PATH,
            folder
        )

        image_files = [

            f

            for f in os.listdir(
                folder_path
            )

            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp"
                )
            )
        ]

        for filename in image_files:

            image_path = os.path.join(
                folder_path,
                filename
            )

            try:

                predicted_class, confidence = predict(
                    image_path
                )

                total_known += 1
                fruit_total += 1

                if predicted_class == fruit_name:

                    correct_known += 1
                    fruit_correct += 1

                    known_correct_confidences.append(
                        confidence
                    )

                else:

                    incorrect_known += 1

                    known_incorrect_confidences.append(
                        confidence
                    )

            except Exception as e:

                print(
                    f"ERROR: {image_path}"
                )

                print(e)

    print(
        f"{fruit_name:<10} "
        f"Total: {fruit_total:<4} "
        f"Correct: {fruit_correct:<4} "
        f"Accuracy: "
        f"{(fruit_correct / fruit_total * 100):.2f}%"
    )


print()

print(
    f"Total known validation images : {total_known}"
)

print(
    f"Correct fruit predictions      : {correct_known}"
)

print(
    f"Incorrect fruit predictions    : {incorrect_known}"
)

print()


# ============================================================
# KNOWN CONFIDENCE DISTRIBUTION
# ============================================================

known_correct_confidences = np.array(
    known_correct_confidences
)

print("=" * 80)
print("KNOWN CORRECT CONFIDENCE DISTRIBUTION")
print("=" * 80)

print(
    f"Minimum : "
    f"{np.min(known_correct_confidences):.2f}%"
)

print(
    f"Maximum : "
    f"{np.max(known_correct_confidences):.2f}%"
)

print(
    f"Mean    : "
    f"{np.mean(known_correct_confidences):.2f}%"
)

print(
    f"5th pct : "
    f"{np.percentile(known_correct_confidences, 5):.2f}%"
)

print(
    f"10th pct: "
    f"{np.percentile(known_correct_confidences, 10):.2f}%"
)

print(
    f"25th pct: "
    f"{np.percentile(known_correct_confidences, 25):.2f}%"
)

print(
    f"50th pct: "
    f"{np.percentile(known_correct_confidences, 50):.2f}%"
)

print(
    f"75th pct: "
    f"{np.percentile(known_correct_confidences, 75):.2f}%"
)

print(
    f"90th pct: "
    f"{np.percentile(known_correct_confidences, 90):.2f}%"
)

print(
    f"95th pct: "
    f"{np.percentile(known_correct_confidences, 95):.2f}%"
)

print()


# ============================================================
# TEST OOD IMAGES
# ============================================================

print("=" * 80)
print("TESTING OOD IMAGES")
print("=" * 80)

ood_results = []

ood_files = [

    f

    for f in os.listdir(
        OOD_PATH
    )

    if f.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        )
    )
]

ood_files.sort()


for filename in ood_files:

    image_path = os.path.join(
        OOD_PATH,
        filename
    )

    try:

        predicted_class, confidence = predict(
            image_path
        )

        ood_results.append({
            "filename": filename,
            "prediction": predicted_class,
            "confidence": confidence
        })

        print(
            f"{filename:<35} "
            f"{predicted_class:<8} "
            f"{confidence:>6.2f}%"
        )

    except Exception as e:

        print(
            f"ERROR: {filename}"
        )

        print(e)


# ============================================================
# OOD CONFIDENCE DISTRIBUTION
# ============================================================

ood_confidences = np.array([

    result["confidence"]

    for result in ood_results

])


print()

print("=" * 80)
print("OOD CONFIDENCE DISTRIBUTION")
print("=" * 80)

print(
    f"Minimum : "
    f"{np.min(ood_confidences):.2f}%"
)

print(
    f"Maximum : "
    f"{np.max(ood_confidences):.2f}%"
)

print(
    f"Mean    : "
    f"{np.mean(ood_confidences):.2f}%"
)

print(
    f"25th pct: "
    f"{np.percentile(ood_confidences, 25):.2f}%"
)

print(
    f"50th pct: "
    f"{np.percentile(ood_confidences, 50):.2f}%"
)

print(
    f"75th pct: "
    f"{np.percentile(ood_confidences, 75):.2f}%"
)

print()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

print()

print(
    "Threshold | Known Accepted | Known Rejected | "
    "OOD Rejected | OOD Rejection Rate"
)

print("-" * 80)


candidate_thresholds = [
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95
]


for threshold in candidate_thresholds:

    known_accepted = np.sum(
        known_correct_confidences >= threshold
    )

    known_rejected = (
        len(known_correct_confidences)
        - known_accepted
    )

    ood_rejected = np.sum(
        ood_confidences < threshold
    )

    ood_total = len(
        ood_confidences
    )

    ood_rejection_rate = (
        ood_rejected /
        ood_total *
        100
    )

    print(
        f"{threshold:>9.0f}% | "
        f"{known_accepted:>14} | "
        f"{known_rejected:>14} | "
        f"{ood_rejected:>12} | "
        f"{ood_rejection_rate:>17.2f}%"
    )


# ============================================================
# KNOWN RETENTION AT OOD REJECTION TARGETS
# ============================================================

print()
print("=" * 80)
print("THRESHOLDS THAT REJECT OOD IMAGES")
print("=" * 80)

for threshold in candidate_thresholds:

    known_acceptance_rate = (

        np.sum(
            known_correct_confidences >= threshold
        )
        /
        len(
            known_correct_confidences
        )
        *
        100
    )

    ood_rejection_rate = (

        np.sum(
            ood_confidences < threshold
        )
        /
        len(
            ood_confidences
        )
        *
        100
    )

    print(
        f"{threshold:>3}% threshold -> "
        f"Known retention: "
        f"{known_acceptance_rate:>6.2f}% | "
        f"OOD rejection: "
        f"{ood_rejection_rate:>6.2f}%"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("CALIBRATION COMPLETE")
print("=" * 80)

print(
    "No model was modified."
)

print(
    "No dataset was modified."
)

print(
    "No threshold was automatically selected."
)

print(
    "Use the threshold table above to determine "
    "whether confidence-based rejection is viable."
)

print("=" * 80)