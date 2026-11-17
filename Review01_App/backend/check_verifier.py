
from pathlib import Path

# Project backend directory
BASE_DIR = Path(__file__).resolve().parent

# Common locations where the verifier might be saved
search_locations = [
    BASE_DIR,
    BASE_DIR / "models",
    BASE_DIR / "model",
    BASE_DIR / "fruit_verifier",
    BASE_DIR / "models" / "fruit_verifier",
    BASE_DIR.parent / "fruit_verifier",
]

# Model file extensions and common checkpoint names
extensions = {".pt", ".pth", ".onnx", ".safetensors"}
known_names = {
    "best.pt",
    "best.pth",
    "model.pt",
    "model.pth",
    "fruit_verifier.pth",
    "fruit_verifier.pt",
    "resnet18.pth",
    "resnet18.pt",
}

print("=" * 55)
print("   RESNET18 FRUIT VERIFIER LOCAL CHECK")
print("=" * 55)
print(f"\nBackend folder: {BASE_DIR}\n")

found = set()

# Search the common locations and their subfolders
for folder in search_locations:
    if folder.exists():
        for file in folder.rglob("*"):
            if file.is_file() and (
                file.suffix.lower() in extensions
                or file.name.lower() in known_names
            ):
                found.add(file.resolve())

if found:
    print("POSSIBLE MODEL FILES FOUND:\n")
    for file in sorted(found):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"File: {file}")
        print(f"Size: {size_mb:.2f} MB")
        print("-" * 55)

    print("\nCHECK COMPLETE!")
    print("Verify which file is the ResNet18 verifier.")
else:
    print("No model checkpoint found in the searched folders.")
    print("The model may be saved elsewhere or only in Colab.")
    print("\nCHECK COMPLETE!")
