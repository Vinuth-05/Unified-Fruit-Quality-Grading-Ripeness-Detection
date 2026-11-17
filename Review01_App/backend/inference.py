import os

from pathlib import Path

from collections import Counter



import torch

import torch.nn as nn

from torchvision import models, transforms

from PIL import Image, ImageDraw, ImageFont

from transformers import AutoImageProcessor, ViTForImageClassification

from ultralytics import YOLO





# ============================================================

# PROJECT CONFIGURATION

# ============================================================



PROJECT_PATH = Path(__file__).resolve().parents[2]



VIT_MODEL_PATH = PROJECT_PATH / "Best_ViT_Model_V2"



BACKEND_DIR = Path(__file__).resolve().parent

YOLO_MODEL_PATH = BACKEND_DIR / "models" / "best_candidate.pt"



VERIFIER_MODEL_PATH = (

    PROJECT_PATH

    / "Fruit_Verifier_Model"

    / "fruit_verifier_resnet18.pth"

)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



CLASS_NAMES = [

    "Apple_Overripe",

    "Apple_Ripe",

    "Apple_Unripe",

    "Orange_Overripe",

    "Orange_Ripe",

    "Orange_Unripe",

    "Pear_Overripe",

    "Pear_Ripe",

    "Pear_Unripe",

]



SUPPORTED_FRUITS = {"Apple", "Orange", "Pear"}



GRADING_RULES = {

    "Ripe": {

        "grade": "Grade A",

        "assessment": "Ripe",

        "recommendation": (

            "Candidate for sale or consumption after normal inspection"

        ),

    },

    "Unripe": {

        "grade": "Grade B",

        "assessment": "Requires further ripening",

        "recommendation": "Hold for further ripening and reassess",

    },

    "Overripe": {

        "grade": "Grade C",

        "assessment": "Overripe",

        "recommendation": (

            "Inspect promptly for damage or spoilage; "

            "consider suitable processing if appropriate"

        ),

    },

}



YOLO_CONFIDENCE = 0.25

YOLO_IOU = 0.45

CROP_PADDING_RATIO = 0.04



MIN_CONFIDENCE = 70.0

MIN_MARGIN = 10.0



RIPENESS_COLORS = {

    "Ripe": (35, 180, 80),

    "Unripe": (245, 190, 30),

    "Overripe": (225, 55, 55),

    "Unknown": (130, 130, 130),

    "Needs Review": (245, 140, 30),

}





# ============================================================

# LOAD VIT V2

# ============================================================



print("Loading ViT V2 model...")



if not VIT_MODEL_PATH.exists():

    raise FileNotFoundError(

        f"ViT model folder not found: {VIT_MODEL_PATH}"

    )



processor = AutoImageProcessor.from_pretrained(str(VIT_MODEL_PATH))

model = ViTForImageClassification.from_pretrained(str(VIT_MODEL_PATH))

model.to(DEVICE)

model.eval()



saved_labels = getattr(model.config, "id2label", None)



if saved_labels:

    try:

        configured_names = [

            saved_labels.get(i, saved_labels.get(str(i)))

            for i in range(len(CLASS_NAMES))

        ]



        if all(configured_names):

            CLASS_NAMES = configured_names



    except (AttributeError, TypeError):

        pass



print("ViT V2 loaded successfully.")

print("Device:", DEVICE)

print("ViT classes:", CLASS_NAMES)





# ============================================================

# LOAD YOLO V3

# ============================================================



print("Loading YOLO V3 detector...")



if not YOLO_MODEL_PATH.exists():

    raise FileNotFoundError(

        f"YOLO model not found: {YOLO_MODEL_PATH}\n"

        "Create backend/models/ and copy your saved best.pt into it."

    )



detector = YOLO(str(YOLO_MODEL_PATH))



print("YOLO V3 loaded successfully.")

print("Detector classes:", detector.names)





# ============================================================

# LOAD RESNET18 FRUIT VERIFIER

# ============================================================



print("Loading ResNet18 fruit verifier...")



if not VERIFIER_MODEL_PATH.exists():

    raise FileNotFoundError(

        f"Fruit verifier not found: {VERIFIER_MODEL_PATH}"

    )



verifier_checkpoint = torch.load(

    VERIFIER_MODEL_PATH,

    map_location=DEVICE,

    weights_only=True,

)



VERIFIER_CLASSES = verifier_checkpoint["classes"]

VERIFIER_IMAGE_SIZE = int(verifier_checkpoint["image_size"])



verifier_model = models.resnet18(weights=None)



verifier_model.fc = nn.Linear(

    verifier_model.fc.in_features,

    len(VERIFIER_CLASSES),

)



verifier_model.load_state_dict(

    verifier_checkpoint["model_state_dict"]

)



verifier_model.to(DEVICE)

verifier_model.eval()



verifier_transform = transforms.Compose([

    transforms.Resize(

        (VERIFIER_IMAGE_SIZE, VERIFIER_IMAGE_SIZE)

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225],

    ),

])



print("ResNet18 verifier loaded successfully.")

print("Verifier classes:", VERIFIER_CLASSES)

print("Verifier image size:", VERIFIER_IMAGE_SIZE)





# ============================================================

# RESNET18 VERIFIER HELPER

# ============================================================



def verify_fruit_crop(image: Image.Image) -> dict:

    """Predict which supported fruit is present in a crop."""



    image = image.convert("RGB")

    inputs = verifier_transform(image).unsqueeze(0).to(DEVICE)



    with torch.inference_mode():

        logits = verifier_model(inputs)

        probabilities = torch.softmax(logits, dim=1)[0]



    best_index = int(torch.argmax(probabilities).item())



    return {

        "fruit_type": VERIFIER_CLASSES[best_index],

        "confidence": round(

            float(probabilities[best_index].item() * 100),

            2,

        ),

    }





# ============================================================

# VIT CLASSIFICATION HELPERS

# ============================================================



def get_prediction_status(confidence: float) -> str:

    if confidence >= 80:

        return "Reliable prediction"



    if confidence >= 60:

        return "Moderate confidence"



    return "Low confidence"





def determine_rejection(confidence: float, margin: float):

    if confidence < MIN_CONFIDENCE:

        return True, "Low prediction confidence"



    if margin < MIN_MARGIN:

        return True, "Low separation between competing classes"



    return False, "Prediction passed rejection check"





def classify_pil_image(image: Image.Image) -> dict:

    """Classify a single fruit crop with the existing ViT V2 model."""



    image = image.convert("RGB")



    inputs = processor(

        images=image,

        return_tensors="pt",

    )



    inputs = {

        key: value.to(DEVICE)

        for key, value in inputs.items()

    }



    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]



    sorted_values, sorted_indices = torch.sort(

        probabilities,

        descending=True,

    )



    predicted_index = int(sorted_indices[0].item())

    confidence = float(sorted_values[0].item() * 100)

    second_confidence = float(sorted_values[1].item() * 100)

    margin = confidence - second_confidence



    predicted_class = CLASS_NAMES[predicted_index]



    confidence_distribution = []



    for rank in range(min(3, len(CLASS_NAMES))):

        index = int(sorted_indices[rank].item())

        value = float(sorted_values[rank].item() * 100)



        confidence_distribution.append({

            "class_name": CLASS_NAMES[index],

            "confidence": round(value, 2),

        })



    is_rejected, rejection_reason = determine_rejection(

        confidence,

        margin,

    )



    if is_rejected:

        return {

            "fruit_type": "Unknown",

            "ripeness": "Unknown",

            "confidence": round(confidence, 2),

            "class_name": predicted_class,

            "quality_grade": "Unknown",

            "quality_assessment": (

                "Unable to reliably classify this fruit crop"

            ),

            "recommendation": "Manual inspection required",

            "prediction_status": "Unknown / Rejected",

            "is_unknown": True,

            "rejection_reason": rejection_reason,

            "top_2_confidence": round(second_confidence, 2),

            "prediction_margin": round(margin, 2),

            "confidence_distribution": confidence_distribution,

        }



    fruit_type, ripeness = predicted_class.split("_", 1)



    grading = GRADING_RULES.get(

        ripeness,

        {

            "grade": "Unknown",

            "assessment": "Unable to determine ripeness",

            "recommendation": "Manual inspection required",

        },

    )



    return {

        "fruit_type": fruit_type,

        "ripeness": ripeness,

        "confidence": round(confidence, 2),

        "class_name": predicted_class,

        "quality_grade": grading["grade"],

        "quality_assessment": grading["assessment"],

        "recommendation": grading["recommendation"],

        "prediction_status": get_prediction_status(confidence),

        "is_unknown": False,

        "rejection_reason": None,

        "top_2_confidence": round(second_confidence, 2),

        "prediction_margin": round(margin, 2),

        "confidence_distribution": confidence_distribution,

    }





# ============================================================

# BACKWARD-COMPATIBLE SINGLE-IMAGE CLASSIFIER

# ============================================================



def predict_image(image_path: str) -> dict:

    """Preserve the original one-image ViT V2 prediction interface."""



    path = Path(image_path)



    if not path.exists():

        raise FileNotFoundError(f"Image not found: {path}")



    with Image.open(path) as source:

        return classify_pil_image(source)





# ============================================================

# IMAGE CROPPING AND ANNOTATION HELPERS

# ============================================================



def crop_with_padding(image: Image.Image, box):

    """Crop a detector box with a small margin."""



    image_width, image_height = image.size

    x1, y1, x2, y2 = [float(value) for value in box]



    pad_x = max(2, (x2 - x1) * CROP_PADDING_RATIO)

    pad_y = max(2, (y2 - y1) * CROP_PADDING_RATIO)



    left = max(0, int(round(x1 - pad_x)))

    top = max(0, int(round(y1 - pad_y)))

    right = min(image_width, int(round(x2 + pad_x)))

    bottom = min(image_height, int(round(y2 + pad_y)))



    if right <= left or bottom <= top:

        return None



    return image.crop((left, top, right, bottom))





def get_font(size=12):

    """Use a readable font for image annotations."""



    for font_name in ("arial.ttf", "DejaVuSans.ttf"):

        try:

            return ImageFont.truetype(font_name, size)

        except OSError:

            continue



    return ImageFont.load_default()





def boxes_overlap(a, b, padding=3):

    """Return True when two label rectangles overlap."""



    return not (

        a[2] + padding <= b[0]

        or b[2] + padding <= a[0]

        or a[3] + padding <= b[1]

        or b[3] + padding <= a[1]

    )





# ============================================================

# V3: DETECT MULTIPLE FRUITS, THEN CLASSIFY EACH CROP

# ============================================================



def predict_image_v3(

    image_path: str,

    output_dir: str | None = None,

    detection_confidence: float = YOLO_CONFIDENCE,

    iou_threshold: float = YOLO_IOU,

) -> dict:

    """

    Detect fruit with YOLO, verify fruit identity with ResNet18,

    and classify ripeness with ViT V2.



    Conflicting fruit identities are marked Needs Review.

    Rejected crops are excluded from accepted grade totals.

    """



    path = Path(image_path)



    if not path.exists():

        raise FileNotFoundError(f"Image not found: {path}")



    with Image.open(path) as source:

        image = source.convert("RGB")



    yolo_results = detector.predict(

        source=image,

        conf=float(detection_confidence),

        iou=float(iou_threshold),

        imgsz=960,

        augment=False,

        verbose=False,

    )



    yolo_result = yolo_results[0]



    detections = []

    drawing_records = []



    if yolo_result.boxes is not None:

        all_boxes = yolo_result.boxes

        for box_index, box in enumerate(all_boxes):

            coords = box.xyxy[0].detach().cpu().tolist()

            detector_class_id = int(box.cls[0].item())

            detector_confidence = float(box.conf[0].item() * 100)



            detected_fruit = detector.names.get(

                detector_class_id,

                str(detector_class_id),

            )



            if detected_fruit not in SUPPORTED_FRUITS:

                continue

            crop = crop_with_padding(image, coords)



            if crop is None:

                continue



            # Verify fruit identity with ResNet18.

            verification = verify_fruit_crop(crop)



            verified_fruit = verification["fruit_type"]

            verifier_confidence = verification["confidence"]



            # Predict ripeness with ViT.

            classification = classify_pil_image(crop)

            predicted_fruit = classification["fruit_type"]



            verifier_mismatch = verified_fruit != detected_fruit

            # Read ViT's raw fruit identity even if its result was rejected.
            raw_vit_fruit = (
                classification["class_name"].split("_", 1)[0]
                if "_" in classification["class_name"]
                else None
            )
            vit_mismatch = (
                raw_vit_fruit in SUPPORTED_FRUITS
                and raw_vit_fruit != detected_fruit
            )

            # Correct a detector disagreement only when both classifiers
            # agree and both predictions pass the confidence checks.
            classifier_agreement = (
                verified_fruit == raw_vit_fruit
                and verified_fruit in SUPPORTED_FRUITS
                and verifier_confidence >= 75.0
                and classification["confidence"] >= MIN_CONFIDENCE
                and classification["prediction_margin"] >= MIN_MARGIN
                and not classification["is_unknown"]
            )

            if classifier_agreement:
                final_fruit = verified_fruit
                mismatch = final_fruit != detected_fruit
                classification["fruit_type"] = final_fruit
                classification["identity_resolved_by_classifiers"] = mismatch
                classification["detector_disagreement"] = mismatch

                if mismatch:
                    consistency_warning = (
                        f"YOLO detected {detected_fruit}; ResNet18 and ViT "
                        f"agree on {final_fruit}. Check the detection box."
                    )
                    classification["quality_assessment"] = (
                        "Fruit identity supported by ResNet18 and ViT; "
                        "YOLO detector disagrees"
                    )
                    classification["recommendation"] = (
                        "Verify the detection box before automated sorting"
                    )
                    classification["prediction_status"] = (
                        "Classifier Agreement - Detector Disagreement"
                    )
                else:
                    consistency_warning = None
            else:
                mismatch = verifier_mismatch or vit_mismatch
                classification["identity_resolved_by_classifiers"] = False
                classification["detector_disagreement"] = mismatch

                if mismatch:
                    consistency_warning = (
                        f"YOLO detected {detected_fruit}; "
                        f"ResNet18 predicted {verified_fruit} "
                        f"({verifier_confidence:.1f}%); "
                        f"ViT predicted {classification['class_name']}. "
                        "Review this crop manually."
                    )
                    classification["quality_grade"] = "Needs Review"
                    classification["quality_assessment"] = (
                        "Fruit identity disagreement between models"
                    )
                    classification["recommendation"] = (
                        "Review the fruit image before assigning a final grade"
                    )
                    classification["prediction_status"] = "Needs Review"
                    classification["is_unknown"] = True
                    classification["rejection_reason"] = (
                        "Fruit identity disagreement between YOLO, "
                        "ResNet18 or ViT"
                    )
                    classification["ripeness"] = "Unknown"
                    classification["fruit_type"] = "Unknown"
                else:
                    consistency_warning = None

            item = {

                "detection_id": len(detections) + 1,

                "bounding_box": {

                    "x1": round(float(coords[0]), 2),

                    "y1": round(float(coords[1]), 2),

                    "x2": round(float(coords[2]), 2),

                    "y2": round(float(coords[3]), 2),

                },

                "detected_fruit_type": detected_fruit,

                "final_fruit_type": classification["fruit_type"],

                "identity_resolved_by_classifiers": classification.get("identity_resolved_by_classifiers", False),

                "detector_disagreement": classification.get("detector_disagreement", False),


                "detection_confidence": round(

                    detector_confidence, 2

                ),

                "verifier_fruit_type": verified_fruit,

                "verifier_confidence": verifier_confidence,

                "verifier_mismatch": verifier_mismatch,

                **classification,

                "fruit_type_mismatch": mismatch,

                "consistency_warning": consistency_warning,

            }



            detections.append(item)

            drawing_records.append((coords, item))



    # ========================================================

    # ANNOTATE IMAGE

    # ========================================================



    image_width, image_height = image.size



    font = get_font(14)

    legend_font = get_font(15)



    row_height = 29

    legend_header_height = 38



    legend_height = (

        legend_header_height

        + max(1, len(drawing_records)) * row_height

        + 12

    )



    annotated = Image.new(

        "RGB",

        (image_width, image_height + legend_height),

        (248, 249, 251),

    )



    annotated.paste(image, (0, 0))

    draw = ImageDraw.Draw(annotated)



    legend_entries = []



    for coords, item in drawing_records:

        x1, y1, x2, y2 = [

            int(round(float(value))) for value in coords

        ]



        x1 = max(0, min(x1, image_width - 1))

        y1 = max(0, min(y1, image_height - 1))

        x2 = max(x1 + 1, min(x2, image_width - 1))

        y2 = max(y1 + 1, min(y2, image_height - 1))



        if item["fruit_type_mismatch"] and item.get(
            "identity_resolved_by_classifiers", False
        ):
            color = RIPENESS_COLORS.get(
                item["ripeness"], RIPENESS_COLORS["Unknown"]
            )
            detail = (
                f'{item["final_fruit_type"]} | {item["ripeness"]} | '
                f'{item["quality_grade"]} | YOLO said '
                f'{item["detected_fruit_type"]} | CHECK BOX'
            )

        elif item["fruit_type_mismatch"]:
            color = RIPENESS_COLORS["Needs Review"]
            detail = (
                f'{item["detected_fruit_type"]} | NEEDS REVIEW | '
                f'ResNet18: {item["verifier_fruit_type"]} '
                f'({item["verifier_confidence"]:.1f}%) | '
                f'ViT: {item["class_name"]} '
                f'({item["confidence"]:.1f}%)'
            )

        elif item["is_unknown"]:

            color = RIPENESS_COLORS["Unknown"]



            detail = (

                f'{item["detected_fruit_type"]} | UNKNOWN | '

                f'Detection {item["detection_confidence"]:.1f}% | '

                f'ViT {item["confidence"]:.1f}%'

            )



        else:

            color = RIPENESS_COLORS.get(

                item["ripeness"],

                RIPENESS_COLORS["Unknown"],

            )



            detail = (

                f'{item["fruit_type"]} | {item["ripeness"]} | '

                f'{item["quality_grade"]} | Detection '

                f'{item["detection_confidence"]:.1f}% | '

                f'ViT {item["confidence"]:.1f}%'

            )



        draw.rectangle(

            (x1, y1, x2, y2),

            outline=color,

            width=3,

        )



        marker = str(item["detection_id"])



        try:

            marker_bbox = draw.textbbox(

                (0, 0),

                marker,

                font=font,

            )



            marker_w = marker_bbox[2] - marker_bbox[0] + 10

            marker_h = marker_bbox[3] - marker_bbox[1] + 6



        except AttributeError:

            marker_w, marker_h = 20, 22



        marker_x = max(

            0,

            min(x1, image_width - marker_w),

        )



        marker_y = max(

            0,

            min(y1, image_height - marker_h),

        )



        draw.rectangle(

            (

                marker_x,

                marker_y,

                marker_x + marker_w,

                marker_y + marker_h,

            ),

            fill=color,

        )



        draw.text(

            (marker_x + 5, marker_y + 2),

            marker,

            fill=(15, 15, 15),

            font=font,

        )



        legend_entries.append(

            (item["detection_id"], color, detail)

        )



    legend_y = image_height



    draw.rectangle(

        (0, legend_y, image_width, annotated.height),

        fill=(248, 249, 251),

    )



    draw.text(

        (12, legend_y + 9),

        "DETECTION LEGEND | Number on box matches this row",

        fill=(25, 35, 50),

        font=legend_font,

    )



    for index, (detection_id, color, detail) in enumerate(

        legend_entries

    ):

        y = legend_y + legend_header_height + index * row_height



        draw.rectangle(

            (12, y + 5, 25, y + 18),

            fill=color,

        )



        text = f"{detection_id}. {detail}"



        # Shorten long labels so they fit on one line.

        while (

            text

            and draw.textbbox((0, 0), text, font=legend_font)[2]

            > image_width - 42

        ):

            text = text[:-4] + "..."



        draw.text(

            (34, y + 2),

            text,

            fill=(25, 35, 50),

            font=legend_font,

        )



    # ========================================================

    # SAVE ANNOTATED IMAGE

    # ========================================================



    save_dir = Path(output_dir) if output_dir else path.parent

    save_dir.mkdir(parents=True, exist_ok=True)



    annotated_path = save_dir / f"{path.stem}_v3_annotated.jpg"



    annotated.save(annotated_path, quality=92)



    # ========================================================

    # CALCULATE SUMMARY COUNTS

    # ========================================================



    fruit_counts = Counter()

    ripeness_counts = Counter()

    grade_counts = Counter()



    accepted_count = 0

    rejected_count = 0

    mismatch_count = 0



    for item in detections:

        fruit_counts[item["detected_fruit_type"]] += 1



        if item["is_unknown"]:

            rejected_count += 1

        else:

            accepted_count += 1

            ripeness_counts[item["ripeness"]] += 1

            grade_counts[item["quality_grade"]] += 1



        if item["fruit_type_mismatch"]:

            mismatch_count += 1



    return {

        "model_pipeline": (

            "YOLO11n V3 + ResNet18 verifier + ViT V2"

        ),

        "image_name": path.name,

        "total_detected_fruits": len(detections),

        "fruit_counts": {

            fruit: int(fruit_counts.get(fruit, 0))

            for fruit in ("Apple", "Orange", "Pear")

        },

        "ripeness_counts": {

            state: int(ripeness_counts.get(state, 0))

            for state in ("Ripe", "Unripe", "Overripe")

        },

        "grade_counts": {

            grade: int(grade_counts.get(grade, 0))

            for grade in ("Grade A", "Grade B", "Grade C")

        },

        "accepted_classifications": accepted_count,

        "rejected_classifications": rejected_count,

        "fruit_type_mismatches": mismatch_count,

        "annotated_image_path": str(annotated_path),

        "detections": detections,

    }





# ============================================================

# LOCAL TEST

# ============================================================



if __name__ == "__main__":

    image_path = input(

        "Enter image path for V3 multi-fruit testing: "

    ).strip().strip('"')



    result = predict_image_v3(image_path)



    print("\n" + "=" * 72)

    print("V3 MULTI-FRUIT QUALITY GRADING RESULT")

    print("=" * 72)



    print("Image:", result["image_name"])

    print("Total detected fruits:", result["total_detected_fruits"])

    print("Fruit counts:", result["fruit_counts"])

    print("Ripeness counts:", result["ripeness_counts"])

    print("Grade counts:", result["grade_counts"])

    print("Accepted classifications:", result["accepted_classifications"])

    print("Rejected classifications:", result["rejected_classifications"])

    print("Fruit-type mismatches:", result["fruit_type_mismatches"])

    print("Annotated image:", result["annotated_image_path"])



    for item in result["detections"]:

        print("\n" + "-" * 60)

        print("Detection ID:", item["detection_id"])

        print("YOLO fruit:", item["detected_fruit_type"])

        print(

            "YOLO detection confidence:",

            item["detection_confidence"],

            "%",

        )

        print("ResNet18 fruit:", item["verifier_fruit_type"])

        print(

            "ResNet18 confidence:",

            item["verifier_confidence"],

            "%",

        )

        print("ViT class:", item["class_name"])

        print("Ripeness:", item["ripeness"])

        print(

            "ViT classification confidence:",

            item["confidence"],

            "%",

        )

        print("Grade:", item["quality_grade"])

        print("Prediction status:", item["prediction_status"])



        if item["consistency_warning"]:

            print("WARNING:", item["consistency_warning"])



    print("\nTest completed.")
