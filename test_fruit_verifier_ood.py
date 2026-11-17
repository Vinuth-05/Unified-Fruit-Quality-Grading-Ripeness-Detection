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

UNKNOWN_PATH = r"C:\Users\bdhul\OneDrive\REVIEW_02_IMAGES"


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

print("=" * 75)
print("PHASE 2 - FRUIT VERIFIER OOD TEST")
print("=" * 75)

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_PATH}")
print(f"Images : {UNKNOWN_PATH}")
print()


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Verifier model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(UNKNOWN_PATH):

    raise FileNotFoundError(
        f"Unknown image folder not found:\n{UNKNOWN_PATH}"
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
# LOAD RESNET18
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
    f"Best validation accuracy stored: "
    f"{checkpoint.get('best_val_accuracy', 'N/A')}"
)

print()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_verifier(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
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
        ].item()
        * 100
    )

    probability_dict = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability_dict[class_name] = (
            probabilities[index].item()
            * 100
        )

    return (
        predicted_class,
        confidence,
        probability_dict
    )


# ============================================================
# FIND TEST IMAGES
# ============================================================

files = [
    f
    for f in os.listdir(
        UNKNOWN_PATH
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

files.sort()


if len(files) == 0:

    raise RuntimeError(
        "No image files found."
    )


# ============================================================
# TEST ALL OOD IMAGES
# ============================================================

print("=" * 75)
print("OOD / UNKNOWN IMAGE TEST")
print("=" * 75)

results = []

for filename in files:

    image_path = os.path.join(
        UNKNOWN_PATH,
        filename
    )

    try:

        (
            predicted_class,
            confidence,
            probabilities
        ) = predict_verifier(
            image_path
        )

        results.append({
            "filename": filename,
            "prediction": predicted_class,
            "confidence": confidence,
            "apple": probabilities["Apple"],
            "orange": probabilities["Orange"],
            "pear": probabilities["Pear"]
        })

        print("-" * 75)

        print(
            f"Image       : {filename}"
        )

        print(
            f"Prediction  : {predicted_class}"
        )

        print(
            f"Confidence  : {confidence:.2f}%"
        )

        print(
            f"Apple       : {probabilities['Apple']:.2f}%"
        )

        print(
            f"Orange      : {probabilities['Orange']:.2f}%"
        )

        print(
            f"Pear        : {probabilities['Pear']:.2f}%"
        )

    except Exception as e:

        print("-" * 75)

        print(
            f"ERROR: {filename}"
        )

        print(e)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 75)
print("OOD TEST SUMMARY")
print("=" * 75)

print(
    f"Total images tested: {len(results)}"
)

print()


# ============================================================
# CONFIDENCE STATISTICS
# ============================================================

if results:

    confidences = np.array([
        result["confidence"]
        for result in results
    ])

    print(
        f"Minimum verifier confidence : "
        f"{np.min(confidences):.2f}%"
    )

    print(
        f"Maximum verifier confidence : "
        f"{np.max(confidences):.2f}%"
    )

    print(
        f"Mean verifier confidence    : "
        f"{np.mean(confidences):.2f}%"
    )

    print(
        f"25th percentile             : "
        f"{np.percentile(confidences, 25):.2f}%"
    )

    print(
        f"50th percentile             : "
        f"{np.percentile(confidences, 50):.2f}%"
    )

    print(
        f"75th percentile             : "
        f"{np.percentile(confidences, 75):.2f}%"
    )

    print()


# ============================================================
# PREDICTION COUNTS
# ============================================================

prediction_counts = {
    "Apple": 0,
    "Orange": 0,
    "Pear": 0
}

for result in results:

    prediction_counts[
        result["prediction"]
    ] += 1


print("=" * 75)
print("VERIFIER PREDICTION DISTRIBUTION")
print("=" * 75)

for fruit in CLASS_NAMES:

    print(
        f"{fruit:<10}: "
        f"{prediction_counts[fruit]}"
    )


# ============================================================
# IMPORTANT NOTE
# ============================================================

print()
print("=" * 75)
print("IMPORTANT")
print("=" * 75)

print(
    "This test DOES NOT apply an OOD rejection threshold."
)

print(
    "It only measures how the 3-class verifier "
    "responds to unsupported images."
)

print(
    "Do NOT integrate the verifier into the API yet."
)

print("=" * 75)