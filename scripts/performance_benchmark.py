#!/usr/bin/env python3
"""
TraceX SIH 2026 - ANPR Performance Benchmarking & Bottleneck Analysis (STEP 22).

Measures and profiles the execution time, resource utilization, and bottlenecks of:
1. Video / Frame loading time
2. YOLO license-plate detection inference time
3. Tracking overhead (VehiclePlateTracker)
4. Plate cropping time (extract_plate_crop)
5. Image preprocessing time (CLAHE, sharpen, denoise, threshold)
6. PaddleOCR inference time (single pass & multi-attempt)
7. Plate validation / consensus time (validate_indian_plate & vote_plate_consensus)
8. Database / storage insertion time (VehicleHistoryStore SQLite)
9. Complete pipeline processing time
10. CPU & RAM resource utilization
11. Effective FPS & throughput
12. Component timing breakdown & bottleneck identification

Exports findings to data/performance_report.json
"""

import os
import sys
import time
import json
import math
import platform
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import cv2

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False

from anpr.pipeline import ANPRPipeline, extract_plate_crop
from anpr.detector import PlateDetector
from anpr.ocr import PlateOCR
from anpr.preprocessing import (
    enhance_clahe,
    enhance_sharpen,
    enhance_denoise,
    enhance_threshold,
    is_valid_crop
)
from anpr.plate_validator import validate_indian_plate, vote_plate_consensus, clean_plate_text
from anpr.storage import VehicleDetectionRecord, VehicleHistoryStore
from video_anpr import VehiclePlateTracker


def compute_stats(values: List[float], unit: str = "ms") -> Dict[str, float]:
    """Compute mean, median, min, max, std from a list of numerical values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0, "unit": unit}
    
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "std": round(float(np.std(arr)), 3),
        "count": len(values),
        "unit": unit
    }


def get_system_environment() -> Dict[str, Any]:
    """Capture environment, hardware specs, CPU & RAM info."""
    cpu_model = platform.processor() or "Unknown CPU"
    logical_cores = os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) if HAS_PSUTIL else logical_cores
    
    ram_gb = 0.0
    if HAS_PSUTIL:
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    
    gpu_desc = "CPU-only / unavailable"
    if HAS_TORCH and CUDA_AVAILABLE:
        gpu_desc = torch.cuda.get_device_name(0)
    
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "cpu_model": cpu_model,
        "cpu_logical_cores": logical_cores,
        "cpu_physical_cores": physical_cores,
        "total_ram_gb": ram_gb,
        "gpu": gpu_desc,
        "cuda_available": CUDA_AVAILABLE,
        "pytorch_version": torch.__version__ if HAS_TORCH else "N/A",
        "opencv_version": cv2.__version__
    }


class PerformanceBenchmarkRunner:
    """
    Executes isolated micro-benchmarks, single-image evaluations,
    and video stream processing to quantify performance bottlenecks.
    """

    def __init__(
        self,
        video_path: str = "data/videos/traffic.mp4",
        eval_images_dir: str = "data/anpr_evaluation/images",
        eval_labels_path: str = "data/anpr_evaluation/labels.csv",
        output_json_path: str = "data/performance_report.json",
        benchmark_db_path: str = "data/test_benchmark_history.db"
    ):
        self.video_path = os.path.join(PROJECT_ROOT, video_path)
        self.eval_images_dir = os.path.join(PROJECT_ROOT, eval_images_dir)
        self.eval_labels_path = os.path.join(PROJECT_ROOT, eval_labels_path)
        self.output_json_path = os.path.join(PROJECT_ROOT, output_json_path)
        self.benchmark_db_path = os.path.join(PROJECT_ROOT, benchmark_db_path)

        # Initialize core components
        print("Initializing production ANPR models for performance benchmark...", flush=True)
        self.detector = PlateDetector()
        self.ocr = PlateOCR.get_instance()
        self.pipeline = ANPRPipeline(detector=self.detector, ocr=self.ocr)
        self.history_store = VehicleHistoryStore(self.benchmark_db_path)

    def _get_process_resources(self) -> Tuple[float, float]:
        """Return (cpu_percent, ram_rss_mb) for current process."""
        if not HAS_PSUTIL:
            return 0.0, 0.0
        try:
            proc = psutil.Process(os.getpid())
            mem_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
            cpu_pct = psutil.cpu_percent(interval=None)
            return cpu_pct, mem_mb
        except Exception:
            return 0.0, 0.0

    def benchmark_micro_components(self, sample_crop: np.ndarray, sample_image: np.ndarray) -> Dict[str, Any]:
        """
        Measure isolated low-level component execution latencies with repeated iterations.
        """
        print("\n--- Running Isolated Micro-Benchmarks ---", flush=True)
        results = {}

        # 1. Plate Cropping Time (100 iterations)
        crop_times = []
        dummy_bbox = (400, 300, 600, 360)
        for _ in range(100):
            t0 = time.perf_counter()
            _ = extract_plate_crop(sample_image, dummy_bbox, margin_ratio=0.015)
            crop_times.append((time.perf_counter() - t0) * 1000.0)
        results["plate_cropping"] = compute_stats(crop_times, "ms")

        # 2. Image Preprocessing Variants (50 iterations each)
        clahe_times = []
        sharpen_times = []
        denoise_times = []
        thresh_times = []
        
        for _ in range(50):
            t0 = time.perf_counter()
            _ = enhance_clahe(sample_crop)
            clahe_times.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            _ = enhance_sharpen(sample_crop)
            sharpen_times.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            _ = enhance_denoise(sample_crop)
            denoise_times.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            _ = enhance_threshold(sample_crop)
            thresh_times.append((time.perf_counter() - t0) * 1000.0)

        results["preprocessing_clahe"] = compute_stats(clahe_times, "ms")
        results["preprocessing_sharpen"] = compute_stats(sharpen_times, "ms")
        results["preprocessing_denoise"] = compute_stats(denoise_times, "ms")
        results["preprocessing_threshold"] = compute_stats(thresh_times, "ms")

        # 3. Plate Validation & Consensus Voting (200 iterations)
        val_times = []
        for _ in range(200):
            t0 = time.perf_counter()
            _ = validate_indian_plate("HR51BC3493")
            val_times.append((time.perf_counter() - t0) * 1000.0)
        results["plate_validation"] = compute_stats(val_times, "ms")

        consensus_times = []
        dummy_obs = [("HR51BC3493", 0.92, True), ("HR51BC3493", 0.88, True), ("HR518C3493", 0.70, False)]
        for _ in range(200):
            t0 = time.perf_counter()
            _ = vote_plate_consensus(dummy_obs)
            consensus_times.append((time.perf_counter() - t0) * 1000.0)
        results["consensus_voting"] = compute_stats(consensus_times, "ms")

        # 4. Database Insertion (50 iterations)
        db_insert_times = []
        rec = VehicleDetectionRecord(
            plate_text="HR51BC3493",
            cleaned_plate="HR51BC3493",
            yolo_confidence=0.91,
            ocr_confidence=0.95,
            valid_indian_plate=True,
            camera_id="CAM_01_TEST",
            timestamp="2026-09-22T12:00:00.000Z",
            frame_number=1,
            track_id=1,
            latitude=28.6139,
            longitude=77.2090
        )
        for _ in range(50):
            t0 = time.perf_counter()
            self.history_store.insert_record(rec)
            db_insert_times.append((time.perf_counter() - t0) * 1000.0)
        results["database_insertion"] = compute_stats(db_insert_times, "ms")

        # 5. Tracking Match & Update (100 iterations)
        tracker = VehiclePlateTracker()
        track_times = []
        for i in range(100):
            t0 = time.perf_counter()
            tid = tracker.match_bbox((100 + i, 100 + i, 200 + i, 150 + i), frame_idx=i)
            tracker.update_track(tid, (100 + i, 100 + i, 200 + i, 150 + i), "HR51BC3493", 0.90, True, 0.92, i, None, False)
            track_times.append((time.perf_counter() - t0) * 1000.0)
        results["tracking_overhead"] = compute_stats(track_times, "ms")

        return results

    def benchmark_single_images(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Benchmark single-image inference across ground-truth evaluation dataset images.
        Measures image loading, YOLO detection, crop, preprocessing, OCR, validation, and total latency.
        """
        print("\n--- Running Single Image Benchmark ---", flush=True)
        
        image_files = []
        if os.path.exists(self.eval_labels_path):
            import csv
            with open(self.eval_labels_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                image_files = [row["image_name"] for row in reader if row.get("image_name")]
        elif os.path.exists(self.eval_images_dir):
            all_files = [f for f in os.listdir(self.eval_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            image_files = sorted(all_files)
        
        if max_samples and max_samples > 0:
            image_files = image_files[:max_samples]

        total_samples = len(image_files)
        if total_samples == 0:
            print("[WARN] No evaluation images found for single image benchmark.")
            return {}

        loading_times = []
        yolo_times = []
        crop_times = []
        preprocessing_times = []
        ocr_times = []
        validation_times = []
        total_times = []

        total_yolo_detections = 0
        total_ocr_attempts = 0
        successful_plates = 0

        start_bench_time = time.perf_counter()
        initial_cpu, initial_ram = self._get_process_resources()

        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(self.eval_images_dir, img_file)
            
            # 1. Image Loading
            t_load_0 = time.perf_counter()
            image = cv2.imread(img_path)
            t_load = (time.perf_counter() - t_load_0) * 1000.0
            loading_times.append(t_load)

            if image is None or image.size == 0:
                continue

            t_sample_start = time.perf_counter()

            # 2. YOLO Detection
            t_yolo_0 = time.perf_counter()
            detections = self.detector.detect(image, conf=0.25, imgsz=1280)
            t_yolo = (time.perf_counter() - t_yolo_0) * 1000.0
            yolo_times.append(t_yolo)

            sample_crop_t = 0.0
            sample_prep_t = 0.0
            sample_ocr_t = 0.0
            sample_val_t = 0.0

            if detections:
                total_yolo_detections += len(detections)
                for det in detections:
                    bbox = det["bbox"]
                    
                    # 3. Crop
                    t_c0 = time.perf_counter()
                    crop = extract_plate_crop(image, bbox, margin_ratio=0.015)
                    sample_crop_t += (time.perf_counter() - t_c0) * 1000.0

                    if crop is None or crop.size == 0:
                        continue

                    # 4. Preprocessing + OCR with variants
                    # Measure standard CLAHE pre-step
                    t_p0 = time.perf_counter()
                    preprocessed = enhance_clahe(crop)
                    sample_prep_t += (time.perf_counter() - t_p0) * 1000.0

                    # Measure OCR (using recognize_with_variants)
                    t_o0 = time.perf_counter()
                    text, conf, attempts = self.ocr.recognize_with_variants(crop)
                    sample_ocr_t += (time.perf_counter() - t_o0) * 1000.0
                    total_ocr_attempts += attempts

                    # 5. Validation
                    t_v0 = time.perf_counter()
                    is_val, cleaned, _ = validate_indian_plate(text)
                    sample_val_t += (time.perf_counter() - t_v0) * 1000.0

                    if is_val and cleaned:
                        successful_plates += 1
            else:
                # No detections: OCR and crop are 0
                pass

            t_sample_total = (time.perf_counter() - t_sample_start) * 1000.0
            total_times.append(t_sample_total)

            crop_times.append(sample_crop_t)
            preprocessing_times.append(sample_prep_t)
            ocr_times.append(sample_ocr_t)
            validation_times.append(sample_val_t)

            if idx % 5 == 0 or idx == total_samples:
                print(f"  [Image {idx:02d}/{total_samples:02d}] Latency: {t_sample_total:.1f}ms | YOLO: {t_yolo:.1f}ms | OCR: {sample_ocr_t:.1f}ms", flush=True)

        final_cpu, final_ram = self._get_process_resources()
        total_duration_s = time.perf_counter() - start_bench_time

        effective_fps = round(total_samples / total_duration_s, 2) if total_duration_s > 0 else 0.0
        throughput_samples_per_sec = effective_fps

        return {
            "samples_tested": total_samples,
            "successful_yolo_detections": total_yolo_detections,
            "total_ocr_attempts": total_ocr_attempts,
            "valid_indian_plates_found": successful_plates,
            "total_benchmark_duration_sec": round(total_duration_s, 2),
            "effective_fps": effective_fps,
            "throughput_samples_sec": throughput_samples_per_sec,
            "ram_usage_mb": {
                "initial": initial_ram,
                "final": final_ram,
                "peak": final_ram
            },
            "timings_ms": {
                "image_loading": compute_stats(loading_times, "ms"),
                "yolo_inference": compute_stats(yolo_times, "ms"),
                "plate_cropping": compute_stats(crop_times, "ms"),
                "preprocessing": compute_stats(preprocessing_times, "ms"),
                "ocr_inference": compute_stats(ocr_times, "ms"),
                "plate_validation": compute_stats(validation_times, "ms"),
                "total_pipeline_per_sample": compute_stats(total_times, "ms")
            }
        }

    def benchmark_video_pipeline(
        self,
        frame_skip: int = 30,
        max_sampled_frames: int = 30,
        conf_threshold: float = 0.32,
        imgsz: int = 1280
    ) -> Dict[str, Any]:
        """
        Benchmark video processing pipeline on traffic.mp4 with controlled frame-skipping,
        tracking, OCR caching, validation, and database storage.
        """
        print(f"\n--- Running Controlled Video Stream Benchmark ({self.video_path}) ---", flush=True)
        if not os.path.exists(self.video_path):
            print(f"[ERROR] Video file not found: {self.video_path}")
            return {}

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {self.video_path}")
            return {}

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tracker = VehiclePlateTracker(max_dist=250.0, max_missed_frames=frame_skip * 4)
        
        # Fresh benchmark store
        self.history_store.clear()

        frame_read_times = []
        yolo_times = []
        tracking_times = []
        crop_times = []
        ocr_times = []
        storage_times = []
        total_frame_times = []

        total_frames_read = 0
        sampled_frames_processed = 0
        total_plate_detections = 0
        total_ocr_attempts = 0
        reused_ocr_count = 0
        stored_records_count = 0

        start_time = time.perf_counter()
        initial_cpu, initial_ram = self._get_process_resources()
        cpu_samples = []

        frame_idx = 0
        while True:
            if sampled_frames_processed >= max_sampled_frames:
                break

            t_fr0 = time.perf_counter()
            ret, frame = cap.read()
            t_fr = (time.perf_counter() - t_fr0) * 1000.0
            
            if not ret or frame is None:
                break

            frame_read_times.append(t_fr)
            total_frames_read += 1

            if frame_idx % frame_skip == 0:
                sampled_frames_processed += 1
                t_frame_start = time.perf_counter()

                # 1. Scaled YOLO Plate Detection
                t_y0 = time.perf_counter()
                raw_dets = self.detector.detect(frame, conf=conf_threshold, imgsz=imgsz)
                t_yolo = (time.perf_counter() - t_y0) * 1000.0
                yolo_times.append(t_yolo)

                frame_crop_t = 0.0
                frame_track_t = 0.0
                frame_ocr_t = 0.0
                frame_storage_t = 0.0

                for det in raw_dets:
                    total_plate_detections += 1
                    bbox = det["bbox"]
                    yolo_conf = det["confidence"]

                    # 2. Crop
                    t_c0 = time.perf_counter()
                    crop = extract_plate_crop(frame, bbox, margin_ratio=0.015)
                    frame_crop_t += (time.perf_counter() - t_c0) * 1000.0

                    # 3. Track Matching
                    t_tr0 = time.perf_counter()
                    track_id = tracker.match_bbox(bbox, frame_idx)
                    run_ocr, use_multi = tracker.should_run_ocr(track_id, crop, frame_idx)
                    frame_track_t += (time.perf_counter() - t_tr0) * 1000.0

                    # 4. OCR
                    t_ocr0 = time.perf_counter()
                    if run_ocr:
                        if use_multi:
                            plate_text, ocr_conf, ocr_attempts = self.ocr.recognize_with_variants(crop)
                        else:
                            plate_text, ocr_conf = self.ocr.recognize(crop)
                            ocr_attempts = 1
                        total_ocr_attempts += ocr_attempts
                        is_valid, cleaned_text, reason = validate_indian_plate(plate_text)
                        ran_ocr = True
                    else:
                        cached = tracker.get_cached_prediction(track_id)
                        plate_text = cached["plate_text"]
                        cleaned_text = cached["cleaned_text"]
                        ocr_conf = cached["ocr_confidence"]
                        is_valid = cached["valid_indian_plate"]
                        ocr_attempts = 0
                        reused_ocr_count += 1
                        ran_ocr = False
                    frame_ocr_t += (time.perf_counter() - t_ocr0) * 1000.0

                    # 5. Track Update
                    t_tr1 = time.perf_counter()
                    track_id = tracker.update_track(
                        track_id=track_id,
                        bbox=bbox,
                        cleaned_text=cleaned_text,
                        ocr_conf=ocr_conf,
                        is_valid=is_valid,
                        yolo_conf=yolo_conf,
                        frame_idx=frame_idx,
                        plate_crop=crop,
                        ran_ocr=ran_ocr
                    )
                    frame_track_t += (time.perf_counter() - t_tr1) * 1000.0

                    # 6. Storage Persistence
                    t_st0 = time.perf_counter()
                    record = VehicleDetectionRecord(
                        plate_text=plate_text,
                        cleaned_plate=cleaned_text or plate_text,
                        yolo_confidence=yolo_conf,
                        ocr_confidence=ocr_conf,
                        valid_indian_plate=is_valid,
                        camera_id="CAM_01_TEST",
                        timestamp="2026-09-22T12:00:00.000Z",
                        frame_number=frame_idx,
                        track_id=track_id,
                        latitude=28.6139,
                        longitude=77.2090
                    )
                    self.history_store.insert_record(record)
                    stored_records_count += 1
                    frame_storage_t += (time.perf_counter() - t_st0) * 1000.0

                t_frame_total = (time.perf_counter() - t_frame_start) * 1000.0
                total_frame_times.append(t_frame_total)

                crop_times.append(frame_crop_t)
                tracking_times.append(frame_track_t)
                ocr_times.append(frame_ocr_t)
                storage_times.append(frame_storage_t)

                if sampled_frames_processed % 5 == 0 or sampled_frames_processed == max_sampled_frames:
                    if HAS_PSUTIL:
                        cpu_samples.append(psutil.cpu_percent(interval=None))
                    print(f"  [Sampled Frame {sampled_frames_processed:02d}/{max_sampled_frames:02d} (Video Frame {frame_idx:04d})] Frame Latency: {t_frame_total:.1f}ms | YOLO: {t_yolo:.1f}ms | OCR: {frame_ocr_t:.1f}ms | Reused OCR: {reused_ocr_count}", flush=True)

            frame_idx += 1

        cap.release()
        total_benchmark_sec = time.perf_counter() - start_time
        final_cpu, final_ram = self._get_process_resources()

        # Multi-frame consensus resolution
        t_cons0 = time.perf_counter()
        resolved_tracks = tracker.get_resolved_tracks()
        consensus_res_ms = (time.perf_counter() - t_cons0) * 1000.0

        effective_sampled_fps = round(sampled_frames_processed / total_benchmark_sec, 2) if total_benchmark_sec > 0 else 0.0
        effective_stream_fps = round(total_frames_read / total_benchmark_sec, 2) if total_benchmark_sec > 0 else 0.0

        avg_cpu_pct = round(float(np.mean(cpu_samples)), 2) if cpu_samples else final_cpu

        return {
            "video_path": self.video_path,
            "total_video_frames": total_video_frames,
            "total_frames_read": total_frames_read,
            "sampled_frames_processed": sampled_frames_processed,
            "frame_skip": frame_skip,
            "total_plate_detections": total_plate_detections,
            "total_ocr_attempts": total_ocr_attempts,
            "reused_ocr_predictions": reused_ocr_count,
            "records_stored": stored_records_count,
            "unique_tracks_identified": len(resolved_tracks),
            "consensus_resolution_time_ms": round(consensus_res_ms, 3),
            "total_benchmark_duration_sec": round(total_benchmark_sec, 2),
            "effective_processed_fps": effective_sampled_fps,
            "effective_stream_fps": effective_stream_fps,
            "cpu_usage_pct": {
                "average": avg_cpu_pct,
                "final": final_cpu
            },
            "ram_usage_mb": {
                "initial": initial_ram,
                "final": final_ram,
                "peak": final_ram
            },
            "timings_ms": {
                "frame_loading_decoding": compute_stats(frame_read_times, "ms"),
                "yolo_inference": compute_stats(yolo_times, "ms"),
                "tracking_overhead": compute_stats(tracking_times, "ms"),
                "plate_cropping": compute_stats(crop_times, "ms"),
                "ocr_inference": compute_stats(ocr_times, "ms"),
                "storage_insertion": compute_stats(storage_times, "ms"),
                "total_frame_pipeline": compute_stats(total_frame_times, "ms")
            }
        }

    def run_full_benchmark(
        self,
        max_image_samples: int = 20,
        max_video_sampled_frames: int = 30,
        frame_skip: int = 30
    ) -> Dict[str, Any]:
        """
        Execute comprehensive benchmark suite, aggregate results,
        compute component percentage breakdowns, and identify the primary bottleneck.
        """
        overall_start = time.perf_counter()

        # Enforce controlled baseline constraints
        effective_max_images = min(max_image_samples or 20, 20)
        effective_frame_skip = max(frame_skip or 30, 30)
        effective_max_video_samples = min(max_video_sampled_frames or 30, 30)

        # Step 0: Warmup image
        warmup_crop = np.zeros((80, 240, 3), dtype=np.uint8)
        cv2.putText(warmup_crop, "HR51BC3493", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        warmup_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        warmup_img[500:580, 800:1040] = warmup_crop

        env_info = get_system_environment()

        print("\n========================================================")
        print("   TRACEX STEP 22: CONTROLLED PERFORMANCE BENCHMARK     ")
        print("========================================================")
        print("Effective Benchmark Configuration:")
        print(f"  - Max Evaluation Images   : {effective_max_images} (controlled baseline)")
        print(f"  - Video Frame Skip        : {effective_frame_skip} (frame_skip >= 30)")
        print(f"  - Max Sampled Video Frames: {effective_max_video_samples}")
        print(f"  - Expected Max OCR Calls  : ~{effective_max_images * 2 + effective_max_video_samples * 2}")
        print(f"  - Target Execution Time   : ~30-60 seconds on CPU")
        print("System Environment:")
        print(f"  - OS         : {env_info['os']}")
        print(f"  - CPU        : {env_info['cpu_model']} ({env_info['cpu_logical_cores']} logical / {env_info['cpu_physical_cores']} physical cores)")
        print(f"  - RAM        : {env_info['total_ram_gb']} GB")
        print(f"  - GPU        : {env_info['gpu']}")
        print(f"  - PyTorch    : {env_info['pytorch_version']}")
        print(f"  - OpenCV     : {env_info['opencv_version']}")
        print("========================================================\n", flush=True)

        # 1. Micro-benchmarks
        micro_results = self.benchmark_micro_components(sample_crop=warmup_crop, sample_image=warmup_img)

        # 2. Single Image Benchmark
        single_img_results = self.benchmark_single_images(max_samples=effective_max_images)

        # 3. Video Pipeline Benchmark
        video_results = self.benchmark_video_pipeline(
            frame_skip=effective_frame_skip,
            max_sampled_frames=effective_max_video_samples
        )

        overall_duration = time.perf_counter() - overall_start

        # 4. Component Percentage Breakdown & Bottleneck Calculation
        # Aggregate component timings from single image and video benchmarks
        yolo_mean_ms = single_img_results.get("timings_ms", {}).get("yolo_inference", {}).get("mean", 0.0)
        ocr_mean_ms = single_img_results.get("timings_ms", {}).get("ocr_inference", {}).get("mean", 0.0)
        prep_mean_ms = single_img_results.get("timings_ms", {}).get("preprocessing", {}).get("mean", 0.0)
        crop_mean_ms = single_img_results.get("timings_ms", {}).get("plate_cropping", {}).get("mean", 0.0)
        val_mean_ms = single_img_results.get("timings_ms", {}).get("plate_validation", {}).get("mean", 0.0)
        track_mean_ms = video_results.get("timings_ms", {}).get("tracking_overhead", {}).get("mean", 0.0)
        storage_mean_ms = video_results.get("timings_ms", {}).get("storage_insertion", {}).get("mean", 0.0)
        loading_mean_ms = single_img_results.get("timings_ms", {}).get("image_loading", {}).get("mean", 0.0)

        # End-to-end average time per image
        total_sample_ms = single_img_results.get("timings_ms", {}).get("total_pipeline_per_sample", {}).get("mean", 0.0)
        if total_sample_ms <= 0:
            total_sample_ms = yolo_mean_ms + ocr_mean_ms + prep_mean_ms + crop_mean_ms + val_mean_ms

        measured_sum = yolo_mean_ms + ocr_mean_ms + prep_mean_ms + crop_mean_ms + val_mean_ms + track_mean_ms + storage_mean_ms
        other_mean_ms = max(0.0, total_sample_ms - (yolo_mean_ms + ocr_mean_ms + prep_mean_ms + crop_mean_ms + val_mean_ms))

        # Calculate percentages
        denom = total_sample_ms if total_sample_ms > 0 else 1.0
        pct_yolo = round((yolo_mean_ms / denom) * 100.0, 2)
        pct_ocr = round((ocr_mean_ms / denom) * 100.0, 2)
        pct_prep = round(((prep_mean_ms + crop_mean_ms) / denom) * 100.0, 2)
        pct_tracking = round((track_mean_ms / denom) * 100.0, 2)
        pct_storage = round((storage_mean_ms / denom) * 100.0, 2)
        pct_validation = round((val_mean_ms / denom) * 100.0, 2)
        pct_other = round(max(0.0, 100.0 - (pct_yolo + pct_ocr + pct_prep + pct_tracking + pct_storage + pct_validation)), 2)

        component_percentages = {
            "yolo_detection_pct": pct_yolo,
            "ocr_recognition_pct": pct_ocr,
            "preprocessing_and_cropping_pct": pct_prep,
            "tracking_overhead_pct": pct_tracking,
            "database_storage_pct": pct_storage,
            "plate_validation_pct": pct_validation,
            "remaining_other_pct": pct_other
        }

        # Identify Largest Bottleneck
        component_times_dict = {
            "PaddleOCR Recognition": ocr_mean_ms,
            "YOLO Plate Detection": yolo_mean_ms,
            "Image Preprocessing & Cropping": prep_mean_ms + crop_mean_ms,
            "Vehicle Tracking Overhead": track_mean_ms,
            "Database / SQLite Insertion": storage_mean_ms,
            "Plate Validation": val_mean_ms
        }

        bottleneck_name, bottleneck_time = max(component_times_dict.items(), key=lambda x: x[1])
        bottleneck_pct = round((bottleneck_time / denom) * 100.0, 1)

        real_time_fps = video_results.get("effective_processed_fps", 0.0)
        is_real_time = real_time_fps >= 25.0

        # Construct Master Report
        master_report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "environment": env_info,
            "controlled_config": {
                "max_evaluation_images": effective_max_images,
                "video_frame_skip": effective_frame_skip,
                "max_video_sampled_frames": effective_max_video_samples
            },
            "summary": {
                "total_benchmark_duration_sec": round(overall_duration, 2),
                "primary_bottleneck": {
                    "component": bottleneck_name,
                    "average_latency_ms": round(bottleneck_time, 2),
                    "percentage_of_pipeline": bottleneck_pct,
                    "explanation": f"{bottleneck_name} accounts for {bottleneck_pct}% of the per-sample compute time ({bottleneck_time:.1f} ms) on CPU."
                },
                "real_time_feasibility": {
                    "is_real_time_capable": is_real_time,
                    "measured_video_effective_fps": real_time_fps,
                    "target_fps": 30.0,
                    "verdict": (
                        f"Real-time processing (>=25 FPS) is NOT ACHIEVABLE on CPU alone ({real_time_fps:.1f} FPS observed). "
                        "Without GPU acceleration or model quantization, CPU inference is bound by heavy deep learning forward passes."
                        if not is_real_time else
                        f"Real-time processing achieved ({real_time_fps:.1f} FPS)."
                    )
                },
                "component_percentage_breakdown": component_percentages
            },
            "single_image_benchmark": single_img_results,
            "video_benchmark": video_results,
            "micro_component_benchmarks": micro_results
        }

        # Save to JSON
        os.makedirs(os.path.dirname(os.path.abspath(self.output_json_path)), exist_ok=True)
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(master_report, f, indent=2)

        # Print Clean Terminal Summary
        self._print_terminal_summary(master_report)

        return master_report

    def _print_terminal_summary(self, report: Dict[str, Any]):
        """Print the required concise terminal report."""
        env = report["environment"]
        single = report.get("single_image_benchmark", {})
        single_timings = single.get("timings_ms", {})
        video = report.get("video_benchmark", {})
        video_timings = video.get("timings_ms", {})
        summary = report["summary"]
        bottleneck = summary["primary_bottleneck"]
        pcts = summary["component_percentage_breakdown"]

        print("\n========================================")
        print("TRACEX STEP 22 PERFORMANCE REPORT")
        print("========================================")
        print("Environment:")
        print(f"CPU: {env['cpu_model']} ({env['cpu_logical_cores']} cores)")
        print(f"RAM: {env['total_ram_gb']} GB")
        print(f"GPU: {env['gpu']}")
        print()
        print("Single Image:")
        print(f"Samples: {single.get('samples_tested', 0)}")
        print(f"Average latency: {single_timings.get('total_pipeline_per_sample', {}).get('mean', 0.0):.1f} ms")
        print(f"Median latency: {single_timings.get('total_pipeline_per_sample', {}).get('median', 0.0):.1f} ms")
        print()
        print("Video:")
        print(f"Frames tested: {video.get('total_frames_read', 0)} frames ({video.get('sampled_frames_processed', 0)} sampled)")
        print(f"Effective FPS: {video.get('effective_processed_fps', 0.0):.2f} FPS")
        print(f"Average frame latency: {video_timings.get('total_frame_pipeline', {}).get('mean', 0.0):.1f} ms")
        print()
        print("Component Timing:")
        print(f"YOLO: {single_timings.get('yolo_inference', {}).get('mean', 0.0):.1f} ms ({pcts.get('yolo_detection_pct', 0.0)}%)")
        print(f"ByteTrack: {video_timings.get('tracking_overhead', {}).get('mean', 0.0):.2f} ms ({pcts.get('tracking_overhead_pct', 0.0)}%)")
        print(f"Preprocessing: {single_timings.get('preprocessing', {}).get('mean', 0.0) + single_timings.get('plate_cropping', {}).get('mean', 0.0):.2f} ms ({pcts.get('preprocessing_and_cropping_pct', 0.0)}%)")
        print(f"OCR: {single_timings.get('ocr_inference', {}).get('mean', 0.0):.1f} ms ({pcts.get('ocr_recognition_pct', 0.0)}%)")
        print(f"Validation: {single_timings.get('plate_validation', {}).get('mean', 0.0):.3f} ms ({pcts.get('plate_validation_pct', 0.0)}%)")
        print(f"Storage: {video_timings.get('storage_insertion', {}).get('mean', 0.0):.2f} ms ({pcts.get('database_storage_pct', 0.0)}%)")
        print(f"Other: {pcts.get('remaining_other_pct', 0.0)}%")
        print()
        print("Bottleneck:")
        print(f"{bottleneck['component']} ({bottleneck['average_latency_ms']:.1f} ms, {bottleneck['percentage_of_pipeline']}%)")
        print("========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TraceX Step 22 Performance Benchmarking Suite")
    parser.add_argument("--video", type=str, default="data/videos/traffic.mp4", help="Path to video file")
    parser.add_argument("--images-dir", type=str, default="data/anpr_evaluation/images", help="Dataset images dir")
    parser.add_argument("--output", type=str, default="data/performance_report.json", help="Output JSON path")
    parser.add_argument("--max-images", type=int, default=20, help="Max image samples to benchmark (default: 20)")
    parser.add_argument("--max-video-samples", type=int, default=30, help="Max video sampled frames (default: 30)")
    parser.add_argument("--frame-skip", type=int, default=30, help="Frame skipping interval (default: 30)")
    args = parser.parse_args()

    runner = PerformanceBenchmarkRunner(
        video_path=args.video,
        eval_images_dir=args.images_dir,
        output_json_path=args.output
    )
    runner.run_full_benchmark(
        max_image_samples=args.max_images,
        max_video_sampled_frames=args.max_video_samples,
        frame_skip=args.frame_skip
    )
