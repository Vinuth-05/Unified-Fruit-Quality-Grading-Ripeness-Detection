import os
import shutil
import random
import numpy as np
import torch

from PIL import Image
from torchvision import models, transforms
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# ============================================================
# PATHS
# ============================================================

PROJECT_PATH = r"C:\Users\bdhul\Documents\Fruit_Grading_Project"

DATASET_PATH = os.path.join(
    PROJECT_PATH,
    "Prepared_Dataset"
)

VERIFIER_PATH = os.path.join(
    PROJECT_PATH,
    "Fruit_Verifier_Model"
)

os.makedirs(
    VERIFIER_PATH,
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("PHASE 2 - SUPPORTED FRUIT VERIFIER")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output : {VERIFIER_PATH}")
print()


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CREATE TEMPORARY 3-CLASS DATASET
# ============================================================

TEMP_DATASET = os.path.join(
    PROJECT_PATH,
    "Fruit_Verifier_Dataset"
)

if os.path.exists(TEMP_DATASET):

    print("Removing previous temporary verifier dataset...")

    shutil.rmtree(
        TEMP_DATASET
    )

os.makedirs(
    TEMP_DATASET,
    exist_ok=True
)

FRUITS = [
    "Apple",
    "Orange",
    "Pear"
]

for fruit in FRUITS:

    os.makedirs(
        os.path.join(
            TEMP_DATASET,
            fruit
        ),
        exist_ok=True
    )


# ============================================================
# COLLECT ALL TRAIN + VALIDATION IMAGES
# ============================================================

print("=" * 70)
print("PREPARING 3-CLASS FRUIT DATASET")
print("=" * 70)

total_copied = 0

for split in ["train", "valid"]:

    split_path = os.path.join(
        DATASET_PATH,
        split
    )

    if not os.path.exists(split_path):

        print(
            f"WARNING: {split_path} not found."
        )

        continue

    for class_name in os.listdir(split_path):

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.isdir(class_path):
            continue

        fruit_name = class_name.split("_")[0]

        if fruit_name not in FRUITS:
            continue

        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            )
        ]

        destination = os.path.join(
            TEMP_DATASET,
            fruit_name
        )

        for index, filename in enumerate(image_files):

            source = os.path.join(
                class_path,
                filename
            )

            # Unique filename prevents collisions
            new_filename = (
                f"{split}_{class_name}_"
                f"{index}_{filename}"
            )

            destination_file = os.path.join(
                destination,
                new_filename
            )

            shutil.copy2(
                source,
                destination_file
            )

            total_copied += 1

print(
    f"Total images copied: {total_copied}"
)

print()


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
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


val_transform = transforms.Compose([

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
# DATASET
# ============================================================

full_dataset = ImageFolder(
    TEMP_DATASET,
    transform=train_transform
)

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    f"Classes: {full_dataset.classes}"
)

print(
    f"Total images: {len(full_dataset)}"
)

for class_index, class_name in enumerate(
    full_dataset.classes
):

    count = sum(
        1
        for _, label in full_dataset.samples
        if label == class_index
    )

    print(
        f"{class_name:<10}: {count}"
    )

print()


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

total_size = len(full_dataset)

train_size = int(
    0.80 * total_size
)

val_size = (
    total_size - train_size
)

generator = torch.Generator().manual_seed(
    SEED
)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

# Create separate dataset so validation has
# deterministic preprocessing.

full_dataset_val = ImageFolder(
    TEMP_DATASET,
    transform=val_transform
)

_, val_dataset = random_split(
    full_dataset_val,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"Training images  : {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)

print()


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("LOADING PRETRAINED RESNET18")
print("=" * 70)

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze backbone first
for parameter in model.parameters():

    parameter.requires_grad = False


# Replace classifier
input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    3
)

model = model.to(DEVICE)

print(
    "ResNet18 loaded."
)

print(
    "Only the final classifier is initially trainable."
)

print()


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0
best_epoch = 0

best_model_path = os.path.join(
    VERIFIER_PATH,
    "fruit_verifier_resnet18.pth"
)

print("=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(
    1,
    EPOCHS + 1
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= train_total

    train_accuracy = (
        train_correct /
        train_total
    ) * 100


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total

    val_accuracy = (
        val_correct /
        val_total
    ) * 100


    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    full_dataset.classes,

                "image_size":
                    IMAGE_SIZE,

                "best_val_accuracy":
                    best_val_accuracy,

                "epoch":
                    best_epoch
            },
            best_model_path
        )

        print(
            "  -> Best verifier model saved."
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Best epoch     : {best_epoch}"
)

print(
    f"Best val acc   : {best_val_accuracy:.2f}%"
)

print(
    f"Model saved at : {best_model_path}"
)

print()


# ============================================================
# SAVE CLASS INFORMATION
# ============================================================

class_file = os.path.join(
    VERIFIER_PATH,
    "classes.txt"
)

with open(
    class_file,
    "w",
    encoding="utf-8"
) as f:

    for class_name in full_dataset.classes:

        f.write(
            class_name + "\n"
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("=" * 70)
print("FRUIT VERIFIER TRAINING COMPLETE")
print("=" * 70)

print()
print("Supported fruits:")
print("  Apple")
print("  Orange")
print("  Pear")

print()
print(
    "Phase 1 ViT model was NOT modified."
)

print(
    "Original dataset was NOT modified."
)

print(
    "Verifier model saved separately."
)

print("=" * 70)