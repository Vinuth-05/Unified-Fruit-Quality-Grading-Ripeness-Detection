import os
import torch
from PIL import Image
from torchvision import models, transforms

# ============================================================
# PATHS
# ============================================================

PROJECT_PATH = r"C:\Users\bdhul\Documents\Fruit_Grading_Project"

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "Fruit_Verifier_Model",
    "fruit_verifier_resnet18.pth"
)

TEST_PATH = os.path.join(
    PROJECT_PATH,
    "Prepared_Dataset",
    "test"
)

# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Apple", "Orange", "Pear"]

FRUIT_FOLDER_MAP = {
    "Apple_Overripe": "Apple",
    "Apple_Ripe": "Apple",
    "Apple_Unripe": "Apple",

    "Orange_Overripe": "Orange",
    "Orange_Ripe": "Orange",
    "Orange_Unripe": "Orange",

    "Pear_Overripe": "Pear",
    "Pear_Ripe": "Pear",
    "Pear_Unripe": "Pear"
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("PHASE 2 - FRUIT VERIFIER FINAL TEST-SET EVALUATION")
print("=" * 80)

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_PATH}")
print(f"Test   : {TEST_PATH}")

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading fruit verifier...")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

print("Verifier loaded successfully.")

# ============================================================
# TEST SET
# ============================================================

total = 0
correct = 0

class_total = {
    "Apple": 0,
    "Orange": 0,
    "Pear": 0
}

class_correct = {
    "Apple": 0,
    "Orange": 0,
    "Pear": 0
}

confidence_correct = []
confidence_incorrect = []

print("\n" + "=" * 80)
print("TEST-SET EVALUATION")
print("=" * 80)

# ============================================================
# READ 9 ORIGINAL TEST FOLDERS
# ============================================================

for folder_name, true_fruit in FRUIT_FOLDER_MAP.items():

    folder_path = os.path.join(
        TEST_PATH,
        folder_name
    )

    if not os.path.isdir(folder_path):
        print(f"WARNING: Missing folder: {folder_path}")
        continue

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(
        f"\n{folder_name:<20} -> {true_fruit:<7} "
        f"Images: {len(files)}"
    )

    for filename in sorted(files):

        image_path = os.path.join(
            folder_path,
            filename
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image_tensor = transform(
                image
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                outputs = model(
                    image_tensor
                )

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

            predicted_fruit = CLASS_NAMES[
                predicted_index.item()
            ]

            confidence_value = (
                confidence.item() * 100
            )

            total += 1

            class_total[
                true_fruit
            ] += 1

            if predicted_fruit == true_fruit:

                correct += 1

                class_correct[
                    true_fruit
                ] += 1

                confidence_correct.append(
                    confidence_value
                )

            else:

                confidence_incorrect.append(
                    confidence_value
                )

        except Exception as e:

            print(
                f"ERROR: {filename} -> {e}"
            )

# ============================================================
# FINAL RESULTS
# ============================================================

accuracy = (
    correct / total * 100
    if total > 0
    else 0
)

print("\n" + "=" * 80)
print("FINAL TEST-SET RESULTS")
print("=" * 80)

print(f"Total test images : {total}")
print(f"Correct           : {correct}")
print(f"Incorrect         : {total - correct}")
print(f"Accuracy          : {accuracy:.2f}%")

# ============================================================
# CLASS-WISE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("CLASS-WISE FRUIT RESULTS")
print("=" * 80)

for fruit in CLASS_NAMES:

    total_class = class_total[fruit]
    correct_class = class_correct[fruit]

    class_accuracy = (
        correct_class / total_class * 100
        if total_class > 0
        else 0
    )

    print(
        f"{fruit:<10} "
        f"Total: {total_class:<4} "
        f"Correct: {correct_class:<4} "
        f"Accuracy: {class_accuracy:.2f}%"
    )

# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CONFIDENCE ANALYSIS")
print("=" * 80)

if confidence_correct:

    print(
        f"Correct predictions mean confidence : "
        f"{sum(confidence_correct) / len(confidence_correct):.2f}%"
    )

    print(
        f"Correct predictions minimum         : "
        f"{min(confidence_correct):.2f}%"
    )

    print(
        f"Correct predictions maximum         : "
        f"{max(confidence_correct):.2f}%"
    )

if confidence_incorrect:

    print(
        f"Incorrect predictions mean confidence: "
        f"{sum(confidence_incorrect) / len(confidence_incorrect):.2f}%"
    )

    print(
        f"Incorrect predictions minimum        : "
        f"{min(confidence_incorrect):.2f}%"
    )

    print(
        f"Incorrect predictions maximum        : "
        f"{max(confidence_incorrect):.2f}%"
    )

# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("PHASE 2 VERIFIER TEST COMPLETE")
print("=" * 80)

print("Original Phase 1 model : NOT MODIFIED")
print("Original dataset       : NOT MODIFIED")
print("Verifier model         : NOT MODIFIED")
print("=" * 80)