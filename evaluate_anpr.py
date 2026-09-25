import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from anpr.pipeline import ANPRPipeline, extract_plate_crop
from anpr.detector import PlateDetector
from anpr.ocr import PlateOCR
from anpr.preprocessing import get_preprocessing_variants, is_valid_crop
from anpr.plate_validator import validate_indian_plate, clean_plate_text


# Built-in ground-truth evaluation dataset (ground-truth annotated from real image assets)
DEFAULT_GROUND_TRUTH_DATASET = {
    "data/images/indian_car.jpg": [
        {
            "plate_number": "HR51BC3493",
            "description": "White Honda City (rear right)",
            "approx_bbox": (770, 665, 845, 695)
        },
        {
            "plate_number": "UP21X3666",
            "description": "Silver Hyundai (rear left)",
            "approx_bbox": (425, 615, 490, 645)
        },
        {
            "plate_number": "DL3CC8387",
            "description": "Black Sedan (center foreground)",
            "approx_bbox": (965, 540, 1045, 570)
        }
    ],
    "data/plates/plate_1.jpg": [
        {
            "plate_number": "HR51BC3493",
            "description": "Crop HR51BC3493",
            "approx_bbox": (0, 0, 64, 16)
        }
    ],
    "data/plates/plate_2.jpg": [
        {
            "plate_number": "UP21X3666",
            "description": "Crop UP21X3666",
            "approx_bbox": (0, 0, 56, 16)
        }
    ],
    "data/plates/plate_4.jpg": [
        {
            "plate_number": "DL3CC8387",
            "description": "Crop DL3CC8387",
            "approx_bbox": (0, 0, 70, 20)
        }
    ]
}


def bbox_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def evaluate_dataset(
    dataset_mapping: Dict[str, List[Dict[str, Any]]],
    save_error_crops: bool = True,
    crop_margin: float = 0.015
) -> Dict[str, Any]:
    """
    Evaluate ANPR pipeline against a ground-truth dataset.

    Computes:
    - detection_rate = detected_plates / total_ground_truth * 100
    - exact_accuracy = correct_exact_matches / total_ground_truth * 100
    - OCR accuracy on detected plates
    - Detailed per-item predictions and failure diagnostic analysis.
    """
    error_dir = "data/plates/error_analysis"
    if save_error_crops:
        os.makedirs(error_dir, exist_ok=True)

    print("=" * 66)
    print("        VEHICLE MONITORING - ANPR ACCURACY & ERROR EVALUATION")
    print("=" * 66)

    pipeline = ANPRPipeline()
    ocr = PlateOCR.get_instance()

    total_ground_truth_plates = 0
    detected_plates_count = 0
    correct_ocr_recognitions = 0
    exact_match_recognitions = 0
    rejected_predictions = 0

    results_table = []
    error_cases = []

    start_eval_time = time.time()

    for image_path, ground_truths in dataset_mapping.items():
        if not os.path.exists(image_path):
            print(f"Warning: File not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        is_crop_file = ("data/plates/" in image_path or "plates" in image_path) and ("video" not in image_path)
        h_img, w_img = image.shape[:2]

        print(f"\nEvaluating: {image_path} ({w_img}x{h_img})")

        if is_crop_file:
            # Standalone crop evaluation (already tightly cropped)
            for gt in ground_truths:
                total_ground_truth_plates += 1
                gt_text = gt["plate_number"]
                detected_plates_count += 1

                pred_text, ocr_c, attempts = ocr.recognize_with_variants(image)
                is_val, clean_pred, reason = validate_indian_plate(pred_text)

                is_exact = (clean_pred == gt_text)
                if is_exact:
                    correct_ocr_recognitions += 1
                    exact_match_recognitions += 1
                    status = "CORRECT"
                else:
                    status = "MISMATCH"
                    if not is_val:
                        rejected_predictions += 1
                    error_cases.append({
                        "file": image_path,
                        "expected": gt_text,
                        "predicted_raw": pred_text,
                        "predicted_cleaned": clean_pred,
                        "yolo_conf": 1.0,
                        "ocr_conf": ocr_c,
                        "valid_format": is_val,
                        "root_cause": "OCR Character Confusion" if pred_text else "Crop Resolution / Contrast"
                    })

                results_table.append({
                    "image": image_path,
                    "expected": gt_text,
                    "predicted": clean_pred,
                    "yolo_conf": 1.0,
                    "ocr_conf": ocr_c,
                    "status": status,
                    "valid_indian": is_val,
                    "reason": reason
                })

                print(f"  [GT: {gt_text:10}] -> Pred: '{clean_pred:10}' | Conf: {ocr_c:.2f} (Attempts: {attempts}) | {status} ({reason})")

        else:
            # Full Image Scene Evaluation (Detection + Inward Crop + OCR + Validation)
            detections = pipeline.detector.detect(image, conf=0.25)

            for gt in ground_truths:
                total_ground_truth_plates += 1
                gt_text = gt["plate_number"]
                gt_box = gt["approx_bbox"]

                best_det = None
                best_iou = 0.0
                for det in detections:
                    iou = bbox_iou(gt_box, det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det

                if best_det is not None and best_iou > 0.15:
                    detected_plates_count += 1
                    bbox = best_det["bbox"]
                    # Adaptive inward margin for YOLO detection boxes
                    crop = extract_plate_crop(image, bbox, margin_ratio=crop_margin)

                    pred_text, ocr_c, attempts = ocr.recognize_with_variants(crop)
                    is_val, clean_pred, reason = validate_indian_plate(pred_text)

                    is_exact = (clean_pred == gt_text)
                    if is_exact:
                        correct_ocr_recognitions += 1
                        exact_match_recognitions += 1
                        status = "CORRECT"
                    else:
                        status = "MISMATCH"
                        if not is_val:
                            rejected_predictions += 1

                        error_crop_path = os.path.join(error_dir, f"error_{gt_text}_pred_{clean_pred or 'empty'}.jpg")
                        if crop is not None and crop.size > 0:
                            cv2.imwrite(error_crop_path, crop)

                        error_cases.append({
                            "file": image_path,
                            "expected": gt_text,
                            "predicted_raw": pred_text,
                            "predicted_cleaned": clean_pred,
                            "yolo_conf": best_det["confidence"],
                            "ocr_conf": ocr_c,
                            "valid_format": is_val,
                            "root_cause": f"OCR/Validation Rejection ({reason})" if not is_val else "OCR Character Substitution",
                            "crop_path": error_crop_path
                        })

                    results_table.append({
                        "image": image_path,
                        "expected": gt_text,
                        "predicted": clean_pred,
                        "yolo_conf": best_det["confidence"],
                        "ocr_conf": ocr_c,
                        "status": status,
                        "valid_indian": is_val,
                        "reason": reason
                    })

                    print(f"  [GT: {gt_text:10}] -> Pred: '{clean_pred:10}' | YOLO: {best_det['confidence']:.2f} | OCR: {ocr_c:.2f} | {status} ({reason})")

                else:
                    status = "NOT DETECTED"
                    error_cases.append({
                        "file": image_path,
                        "expected": gt_text,
                        "predicted_raw": "",
                        "predicted_cleaned": "",
                        "yolo_conf": 0.0,
                        "ocr_conf": 0.0,
                        "valid_format": False,
                        "root_cause": "YOLO Detection Miss (Plate Not Found)"
                    })
                    results_table.append({
                        "image": image_path,
                        "expected": gt_text,
                        "predicted": "NOT DETECTED",
                        "yolo_conf": 0.0,
                        "ocr_conf": 0.0,
                        "status": status,
                        "valid_indian": False,
                        "reason": "Detection Miss"
                    })
                    print(f"  [GT: {gt_text:10}] -> NOT DETECTED by YOLO")

    total_eval_time = time.time() - start_eval_time

    # Calculate Metrics
    detection_rate = (detected_plates_count / total_ground_truth_plates * 100.0) if total_ground_truth_plates > 0 else 0.0
    ocr_accuracy = (correct_ocr_recognitions / detected_plates_count * 100.0) if detected_plates_count > 0 else 0.0
    exact_accuracy = (exact_match_recognitions / total_ground_truth_plates * 100.0) if total_ground_truth_plates > 0 else 0.0

    print("\n" + "=" * 66)
    print("                    EVALUATION METRICS SUMMARY")
    print("=" * 66)
    print(f"Total Ground-Truth Plates Tested : {total_ground_truth_plates}")
    print(f"Plates Successfully Detected     : {detected_plates_count} ({detection_rate:.1f}% Detection Rate)")
    print(f"Correct Exact OCR Recognitions   : {exact_match_recognitions} ({exact_accuracy:.1f}% Exact Accuracy)")
    print(f"OCR Accuracy (on Detections)     : {ocr_accuracy:.1f}%")
    print(f"Invalid / Rejected Predictions   : {rejected_predictions}")
    print(f"Total Evaluation Time            : {total_eval_time:.2f} sec ({total_eval_time / total_ground_truth_plates if total_ground_truth_plates > 0 else 0:.2f}s/item)")
    print("-" * 66)

    if error_cases:
        print("ERROR ANALYSIS & DIAGNOSTICS:")
        for idx, err in enumerate(error_cases, 1):
            print(f"\n  Case #{idx}:")
            print(f"    Source File      : {err['file']}")
            print(f"    Expected Plate   : {err['expected']}")
            print(f"    Predicted Cleaned: '{err['predicted_cleaned']}' (Raw: '{err['predicted_raw']}')")
            print(f"    YOLO Confidence  : {err['yolo_conf']:.2f}")
            print(f"    OCR Confidence   : {err['ocr_conf']:.2f}")
            print(f"    Format Valid     : {err['valid_format']}")
            print(f"    Diagnosed Problem: {err['root_cause']}")
    else:
        print("ERROR ANALYSIS: 0 errors detected across ground-truth test cases.")

    print("=" * 66)
    print("ACCURACY VERIFICATION ASSESSMENT:")
    if total_ground_truth_plates < 30:
        print(f">> Measured Exact Accuracy on available dataset: {exact_accuracy:.1f}% ({exact_match_recognitions}/{total_ground_truth_plates})")
        print(">> Note: With only a small number of local Indian test samples, 90% accuracy cannot yet be statistically verified for general deployment.")
    else:
        print(f">> Exact Accuracy on benchmark dataset: {exact_accuracy:.1f}% ({exact_match_recognitions}/{total_ground_truth_plates})")
        if exact_accuracy >= 90.0:
            print(">> Target 90% accuracy requirement is VERIFIED on this dataset.")
        else:
            print(">> Target 90% accuracy requirement is NOT YET MET.")
    print("=" * 66)

    return {
        "total_gt": total_ground_truth_plates,
        "detected": detected_plates_count,
        "detection_rate": detection_rate,
        "correct_exact": exact_match_recognitions,
        "exact_accuracy": exact_accuracy,
        "ocr_accuracy": ocr_accuracy,
        "eval_time": total_eval_time,
        "results": results_table,
        "errors": error_cases
    }


def load_custom_dataset(dataset_dir: str, annotations_file: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load custom ground-truth dataset from a directory and optional annotations file (JSON / TXT).
    Expected JSON format: {"image_filename.jpg": "HR51BC3493", ...}
    Or filename based: image named 'HR51BC3493_car1.jpg' -> ground truth 'HR51BC3493'.
    """
    dataset = {}
    if annotations_file and os.path.exists(annotations_file):
        with open(annotations_file, "r") as f:
            data = json.load(f)
            for img_name, label in data.items():
                p = os.path.join(dataset_dir, img_name) if dataset_dir else img_name
                if isinstance(label, str):
                    dataset[p] = [{"plate_number": clean_plate_text(label), "description": img_name, "approx_bbox": (0, 0, 9999, 9999)}]
                elif isinstance(label, list):
                    dataset[p] = label
    elif dataset_dir and os.path.exists(dataset_dir):
        for fname in sorted(os.listdir(dataset_dir)):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                p = os.path.join(dataset_dir, fname)
                # Infer plate text if filename is in format 'HR51BC3493_...' or 'UP21X3666.jpg'
                base = os.path.splitext(fname)[0].split("_")[0]
                is_val, clean_label, _ = validate_indian_plate(base)
                if is_val:
                    dataset[p] = [{"plate_number": clean_label, "description": fname, "approx_bbox": (0, 0, 9999, 9999)}]

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VehicleMonitoring ANPR Accuracy & Error Evaluator")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Directory containing test images")
    parser.add_argument("--annotations", type=str, default=None, help="Path to annotations JSON file")
    parser.add_argument("--margin", type=float, default=0.015, help="Crop margin ratio (default: 0.015 / 1.5%)")
    args = parser.parse_args()

    if args.dataset_dir or args.annotations:
        custom_data = load_custom_dataset(args.dataset_dir, args.annotations)
        if custom_data:
            evaluate_dataset(custom_data, crop_margin=args.margin)
        else:
            print(f"Error: No valid annotated images found in {args.dataset_dir or args.annotations}")
    else:
        evaluate_dataset(DEFAULT_GROUND_TRUTH_DATASET, crop_margin=args.margin)
