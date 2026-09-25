#!/usr/bin/env python3
"""
TraceX Large-Scale ANPR Accuracy & Error Evaluation Framework (STEP 21).

Evaluates the end-to-end ANPR pipeline against an independent labeled dataset:
- YOLO Plate Detection Rate
- Exact OCR Recognition Accuracy
- Character-Level Normalized Levenshtein Accuracy
- Invalid/Rejected Rate
- Average OCR Confidence & Processing Latency
- Independent Vehicle vs Repeated Sightings Distinction
- Strict Statistical Sufficiency Verification (>=50 independent vehicles)
- Detailed Failure Classification & Confusion Mapping
- Export to data/anpr_evaluation_report.json
"""

import os
import sys
import time
import json
import csv
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from anpr.pipeline import ANPRPipeline
from anpr.ocr import PlateOCR
from anpr.plate_validator import validate_indian_plate, clean_plate_text


def calculate_character_accuracy(ground_truth: str, predicted: str) -> float:
    """
    Calculate normalized character-level accuracy percentage using Levenshtein distance.
    Formula: max(0.0, 1.0 - (LevenshteinDistance / max(len(gt), len(pred)))) * 100.0
    """
    gt = (ground_truth or "").strip().upper().replace(" ", "")
    pred = (predicted or "").strip().upper().replace(" ", "")

    if not gt and not pred:
        return 100.0
    if not gt or not pred:
        return 0.0

    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    distance = dp[m][n]
    max_len = max(m, n)
    similarity = max(0.0, 1.0 - (distance / max_len))
    return round(similarity * 100.0, 2)


def run_anpr_evaluation(
    dataset_dir: str = "data/anpr_evaluation",
    labels_csv_path: str = "data/anpr_evaluation/labels.csv",
    output_json_path: str = "data/anpr_evaluation_report.json",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute large-scale ANPR evaluation across the labeled dataset.
    """
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(labels_csv_path):
        raise FileNotFoundError(f"Labels file not found: {labels_csv_path}")

    # 1. Load dataset labels
    samples = []
    with open(labels_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image_name", "").strip()
            gt_plate = row.get("ground_truth_plate", "").strip()
            if img_name and gt_plate:
                samples.append((img_name, gt_plate))

    if limit is not None and limit > 0:
        samples = samples[:limit]

    total_samples = len(samples)
    if total_samples == 0:
        raise ValueError("Evaluation dataset is empty. No labeled samples found.")

    # 2. Count independent vehicles and repeated sightings
    plate_counts = Counter(s[1] for s in samples)
    unique_plates = set(plate_counts.keys())
    num_independent_vehicles = len(unique_plates)
    num_repeated_samples = total_samples - num_independent_vehicles

    seen_plate_tracker = defaultdict(int)

    # 3. Initialize ANPR Pipeline (Unmodified production pipeline)
    print("=" * 80)
    print("      TRACEX STEP 21: LARGE-SCALE ANPR ACCURACY EVALUATION")
    print("=" * 80)
    print(f"Dataset Directory         : {images_dir}")
    print(f"Ground-Truth Annotations  : {labels_csv_path}")
    print(f"Total Labeled Samples     : {total_samples}")
    print(f"Unique Independent Plates : {num_independent_vehicles}")
    print(f"Repeated Sightings        : {num_repeated_samples}")
    print("-" * 80)
    print("Initializing production ANPR pipeline...")

    pipeline = ANPRPipeline()

    # Metrics collectors
    successful_yolo_detections = 0
    total_ocr_attempts = 0
    exact_plate_matches = 0
    total_char_accuracy = 0.0
    rejected_invalid_count = 0
    ocr_confidences = []
    processing_times = []

    # Failure categories
    failure_breakdown = {
        "plate_not_detected": 0,
        "ocr_failed": 0,
        "ocr_incorrect": 0,
        "invalid_plate_rejected": 0,
        "preprocessing_failure": 0
    }

    sample_results = []
    error_cases = []
    unique_plate_results = defaultdict(list)

    total_start_time = time.perf_counter()

    for idx, (img_name, gt_plate) in enumerate(samples, 1):
        img_path = os.path.join(images_dir, img_name)
        seen_plate_tracker[gt_plate] += 1
        is_repeated = (seen_plate_tracker[gt_plate] > 1)

        t0 = time.perf_counter()

        # Check image existence & read
        if not os.path.exists(img_path):
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            failure_breakdown["preprocessing_failure"] += 1
            error_cases.append({
                "image_name": img_name,
                "ground_truth": gt_plate,
                "predicted": "FILE_NOT_FOUND",
                "error_type": "preprocessing failure",
                "details": f"File does not exist: {img_path}"
            })
            sample_results.append({
                "image_name": img_name,
                "ground_truth_plate": gt_plate,
                "is_repeated": is_repeated,
                "plate_detected": False,
                "predicted_plate": "",
                "raw_ocr_text": "",
                "yolo_confidence": 0.0,
                "ocr_confidence": 0.0,
                "ocr_attempts": 0,
                "valid_format": False,
                "exact_match": False,
                "character_accuracy_pct": 0.0,
                "error_type": "preprocessing failure",
                "processing_time_ms": round(t_elapsed, 2)
            })
            print(f"[{idx:02d}/{total_samples:02d}] {img_name:<30} | GT: {gt_plate:<12} -> ERROR: Image not found")
            continue

        image = cv2.imread(img_path)
        if image is None or image.size == 0:
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            failure_breakdown["preprocessing_failure"] += 1
            error_cases.append({
                "image_name": img_name,
                "ground_truth": gt_plate,
                "predicted": "CORRUPTED_IMAGE",
                "error_type": "preprocessing failure",
                "details": "cv2.imread failed to decode image"
            })
            sample_results.append({
                "image_name": img_name,
                "ground_truth_plate": gt_plate,
                "is_repeated": is_repeated,
                "plate_detected": False,
                "predicted_plate": "",
                "raw_ocr_text": "",
                "yolo_confidence": 0.0,
                "ocr_confidence": 0.0,
                "ocr_attempts": 0,
                "valid_format": False,
                "exact_match": False,
                "character_accuracy_pct": 0.0,
                "error_type": "preprocessing failure",
                "processing_time_ms": round(t_elapsed, 2)
            })
            print(f"[{idx:02d}/{total_samples:02d}] {img_name:<30} | GT: {gt_plate:<12} -> ERROR: Corrupted image")
            continue

        # Process through full ANPR pipeline
        detections = pipeline.process_frame(image)
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        processing_times.append(t_elapsed)

        if not detections:
            # Failure: plate not detected by YOLO
            failure_breakdown["plate_not_detected"] += 1
            error_cases.append({
                "image_name": img_name,
                "ground_truth": gt_plate,
                "predicted": "[NO_DETECTION]",
                "error_type": "plate not detected",
                "details": "YOLO detector did not identify a license plate bounding box"
            })
            res_entry = {
                "image_name": img_name,
                "ground_truth_plate": gt_plate,
                "is_repeated": is_repeated,
                "plate_detected": False,
                "predicted_plate": "",
                "raw_ocr_text": "",
                "yolo_confidence": 0.0,
                "ocr_confidence": 0.0,
                "ocr_attempts": 0,
                "valid_format": False,
                "exact_match": False,
                "character_accuracy_pct": 0.0,
                "error_type": "plate not detected",
                "processing_time_ms": round(t_elapsed, 2)
            }
            sample_results.append(res_entry)
            unique_plate_results[gt_plate].append(False)
            print(f"[{idx:02d}/{total_samples:02d}] {img_name:<30} | GT: {gt_plate:<12} -> [NO DETECT] ({t_elapsed:.1f}ms)")
            continue

        # YOLO Detection Succeeded
        successful_yolo_detections += 1

        # Match to best detection (matching gt or highest character accuracy)
        best_det = None
        best_match_score = -1.0
        for det in detections:
            pred_cand = det.get("cleaned_text") or det.get("plate_text", "")
            sim = calculate_character_accuracy(gt_plate, pred_cand)
            if sim > best_match_score:
                best_match_score = sim
                best_det = det

        raw_text = best_det.get("plate_text", "").strip()
        cleaned_text = best_det.get("cleaned_text", "").strip()
        yolo_c = float(best_det.get("yolo_confidence", 0.0))
        ocr_c = float(best_det.get("ocr_confidence", 0.0))
        attempts = int(best_det.get("ocr_attempts", 1))
        valid_format = bool(best_det.get("valid_indian_plate", False))

        total_ocr_attempts += attempts
        ocr_confidences.append(ocr_c)

        # Accuracy calculations
        # Check clean match or raw text without spaces
        is_exact = (
            cleaned_text.upper().replace(" ", "") == gt_plate.upper().replace(" ", "")
            or raw_text.upper().replace(" ", "") == gt_plate.upper().replace(" ", "")
        )

        char_acc = calculate_character_accuracy(gt_plate, cleaned_text or raw_text)
        total_char_accuracy += char_acc

        error_type = None
        if is_exact:
            exact_plate_matches += 1
            unique_plate_results[gt_plate].append(True)
            status_str = "EXACT MATCH"
        else:
            unique_plate_results[gt_plate].append(False)
            if not raw_text:
                error_type = "OCR failed"
                failure_breakdown["ocr_failed"] += 1
            elif not valid_format:
                error_type = "invalid plate rejected"
                failure_breakdown["invalid_plate_rejected"] += 1
                rejected_invalid_count += 1
            else:
                error_type = "OCR incorrect"
                failure_breakdown["ocr_incorrect"] += 1

            error_cases.append({
                "image_name": img_name,
                "ground_truth": gt_plate,
                "predicted": cleaned_text or raw_text,
                "error_type": error_type,
                "details": f"YOLO: {yolo_c:.2f}, OCR: {ocr_c:.2f}, Valid: {valid_format} ({best_det.get('validation_reason', '')})"
            })
            status_str = f"MISMATCH ({error_type})"

        sample_results.append({
            "image_name": img_name,
            "ground_truth_plate": gt_plate,
            "is_repeated": is_repeated,
            "plate_detected": True,
            "predicted_plate": cleaned_text or raw_text,
            "raw_ocr_text": raw_text,
            "yolo_confidence": round(yolo_c, 4),
            "ocr_confidence": round(ocr_c, 4),
            "ocr_attempts": attempts,
            "valid_format": valid_format,
            "exact_match": is_exact,
            "character_accuracy_pct": char_acc,
            "error_type": error_type,
            "processing_time_ms": round(t_elapsed, 2)
        })

        repeat_tag = "[REPEAT]" if is_repeated else "[NEW]   "
        print(f"[{idx:02d}/{total_samples:02d}] {repeat_tag} {img_name:<26} | GT: {gt_plate:<10} -> Pred: '{cleaned_text or raw_text:<10}' | Acc: {char_acc:5.1f}% | {status_str} ({t_elapsed:.1f}ms)")

    total_eval_duration = time.perf_counter() - total_start_time

    # 4. Overall Aggregate Metrics
    detection_rate_pct = round((successful_yolo_detections / total_samples) * 100.0, 2) if total_samples > 0 else 0.0
    exact_accuracy_pct = round((exact_plate_matches / total_samples) * 100.0, 2) if total_samples > 0 else 0.0
    avg_char_accuracy_pct = round(total_char_accuracy / total_samples, 2) if total_samples > 0 else 0.0
    invalid_rejected_rate_pct = round((failure_breakdown["invalid_plate_rejected"] / total_samples) * 100.0, 2) if total_samples > 0 else 0.0
    avg_ocr_conf = round(float(np.mean(ocr_confidences)), 4) if ocr_confidences else 0.0
    avg_latency_ms = round(float(np.mean(processing_times)), 2) if processing_times else 0.0
    avg_ocr_attempts = round(total_ocr_attempts / successful_yolo_detections, 2) if successful_yolo_detections > 0 else 0.0

    # 5. Independent Vehicle-Level Metrics
    unique_plates_exact = sum(1 for p, res_list in unique_plate_results.items() if any(res_list))
    unique_plate_accuracy_pct = round((unique_plates_exact / num_independent_vehicles) * 100.0, 2) if num_independent_vehicles > 0 else 0.0

    # 6. Statistical Sufficiency Check
    INDEPENDENT_THRESHOLD = 50
    is_statistically_sufficient = (num_independent_vehicles >= INDEPENDENT_THRESHOLD)

    if not is_statistically_sufficient:
        target_status = "NOT YET VERIFIED"
        target_statement = (
            f">=90% generalized accuracy is NOT YET VERIFIED. "
            f"The current dataset contains {num_independent_vehicles} independent vehicles "
            f"(< {INDEPENDENT_THRESHOLD} required for statistical validity). "
            f"To avoid false claims from repeated crops/sightings, testing must be expanded to >=50 independent vehicles."
        )
    else:
        if exact_accuracy_pct >= 90.0:
            target_status = "VERIFIED"
            target_statement = f">=90% generalized accuracy target is VERIFIED ({exact_accuracy_pct:.1f}% on {num_independent_vehicles} independent vehicles)."
        else:
            target_status = "NOT MET"
            target_statement = f">=90% generalized accuracy target is NOT MET ({exact_accuracy_pct:.1f}% on {num_independent_vehicles} independent vehicles)."

    # 7. Identify Biggest Failure Categories
    sorted_failures = sorted(
        [(k, v) for k, v in failure_breakdown.items() if v > 0],
        key=lambda x: -x[1]
    )

    # 8. Construct Output Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_summary": {
            "total_labeled_samples": total_samples,
            "unique_independent_vehicles": num_independent_vehicles,
            "repeated_vehicle_sightings": num_repeated_samples,
            "dataset_directory": images_dir,
            "labels_csv_file": labels_csv_path
        },
        "performance_metrics": {
            "successful_yolo_detections": successful_yolo_detections,
            "detection_rate_pct": detection_rate_pct,
            "ocr_attempts_total": total_ocr_attempts,
            "ocr_attempts_avg_per_detection": avg_ocr_attempts,
            "exact_plate_matches": exact_plate_matches,
            "exact_recognition_accuracy_pct": exact_accuracy_pct,
            "character_level_accuracy_pct": avg_char_accuracy_pct,
            "invalid_rejected_rate_pct": invalid_rejected_rate_pct,
            "average_ocr_confidence": avg_ocr_conf,
            "total_processing_time_sec": round(total_eval_duration, 2),
            "average_processing_time_ms_per_image": avg_latency_ms
        },
        "independent_vehicle_metrics": {
            "unique_plates_count": num_independent_vehicles,
            "unique_plates_correctly_recognized": unique_plates_exact,
            "unique_plates_accuracy_pct": unique_plate_accuracy_pct,
            "samples_per_plate_distribution": dict(plate_counts)
        },
        "failure_breakdown": failure_breakdown,
        "biggest_failure_categories": sorted_failures,
        "generalized_accuracy_verification": {
            "target_accuracy_pct": 90.0,
            "independent_vehicles_threshold": INDEPENDENT_THRESHOLD,
            "independent_vehicles_available": num_independent_vehicles,
            "is_statistically_sufficient": is_statistically_sufficient,
            "status": target_status,
            "statement": target_statement
        },
        "confusion_and_error_cases": error_cases,
        "sample_evaluations": sample_results
    }

    # Save to JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, mode="w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Terminal Report
    print("\n" + "=" * 80)
    print("                    EVALUATION METRICS & ACCURACY SUMMARY")
    print("=" * 80)
    print("DATASET OVERVIEW:")
    print(f"  Total Labeled Samples Tested   : {total_samples}")
    print(f"  Unique Independent Vehicles    : {num_independent_vehicles}")
    print(f"  Repeated Vehicle Sightings     : {num_repeated_samples}")
    print("-" * 80)
    print("RECOGNITION PERFORMANCE:")
    print(f"  Successful YOLO Detections     : {successful_yolo_detections}/{total_samples} ({detection_rate_pct:.1f}% Detection Rate)")
    print(f"  Exact Plate Matches            : {exact_plate_matches}/{total_samples} ({exact_accuracy_pct:.1f}% Exact Accuracy)")
    print(f"  Character-Level Accuracy       : {avg_char_accuracy_pct:.1f}%")
    print(f"  Invalid / Rejected Rate        : {invalid_rejected_rate_pct:.1f}%")
    print(f"  Average OCR Confidence         : {avg_ocr_conf:.3f}")
    print(f"  Average OCR Attempts           : {avg_ocr_attempts:.2f} per detected plate")
    print(f"  Processing Latency             : {avg_latency_ms:.1f} ms/sample (Total: {total_eval_duration:.2f}s)")
    print("-" * 80)
    print("INDEPENDENT VEHICLE ACCURACY (Deduplicated):")
    print(f"  Unique Plates Correctly Read   : {unique_plates_exact}/{num_independent_vehicles} ({unique_plate_accuracy_pct:.1f}%)")
    print("-" * 80)
    print("FAILURE BREAKDOWN & CATEGORIZATION:")
    for cat, cnt in failure_breakdown.items():
        pct = (cnt / total_samples * 100.0) if total_samples > 0 else 0.0
        print(f"  - {cat.replace('_', ' ').title():<28}: {cnt:3d} ({pct:5.1f}%)")
    print("-" * 80)

    if error_cases:
        print("CONFUSION / ERROR EXAMPLES (Ground Truth -> Predicted -> Error Type):")
        for i, err in enumerate(error_cases[:10], 1):
            print(f"  [{i:02d}] {err['ground_truth']:<12} -> '{err['predicted']:<12}' | {err['error_type']} | {err.get('details', '')}")
        if len(error_cases) > 10:
            print(f"  ... and {len(error_cases) - 10} more error cases (see {output_json_path})")
    else:
        print("CONFUSION / ERROR EXAMPLES: 0 errors detected across the dataset.")

    print("=" * 80)
    print("GENERALIZED ACCURACY TARGET ASSESSMENT:")
    print(f"  Target Accuracy Target         : >= 90.0%")
    print(f"  Independent Vehicles Required  : >= {INDEPENDENT_THRESHOLD}")
    print(f"  Independent Vehicles Evaluated : {num_independent_vehicles}")
    print(f"  Verification Status            : {target_status}")
    print(f"  Details                        : {target_statement}")
    print(f"  Report Exported To             : {output_json_path}")
    print("=" * 80 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TraceX Large-Scale ANPR Accuracy Evaluator")
    parser.add_argument("--dataset-dir", type=str, default="data/anpr_evaluation", help="Dataset directory")
    parser.add_argument("--labels", type=str, default="data/anpr_evaluation/labels.csv", help="Labels CSV file")
    parser.add_argument("--output", type=str, default="data/anpr_evaluation_report.json", help="Output JSON report path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to test")
    args = parser.parse_args()

    run_anpr_evaluation(
        dataset_dir=args.dataset_dir,
        labels_csv_path=args.labels,
        output_json_path=args.output,
        limit=args.limit
    )
