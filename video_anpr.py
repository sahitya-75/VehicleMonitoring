import os
import sys
import time
import math
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2

# Ensure project root is in sys.path when running video_anpr.py directly
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime, timedelta
from anpr.pipeline import ANPRPipeline, extract_plate_crop
from anpr.plate_validator import vote_plate_consensus, validate_indian_plate
from anpr.storage import VehicleDetectionRecord, VehicleHistoryStore


class VehiclePlateTracker:
    """
    Lightweight multi-frame vehicle tracker that aggregates plate observations,
    reuses high-confidence OCR predictions, and performs consensus voting.
    """

    def __init__(self, max_dist: float = 250.0, max_missed_frames: int = 60):
        self.max_dist = max_dist
        self.max_missed_frames = max_missed_frames
        self.tracks = {}  # track_id -> dict
        self.next_track_id = 1

    def match_bbox(self, bbox: Tuple[int, int, int, int], frame_idx: int) -> int:
        """
        Find an existing active track that spatially matches the bounding box.
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        best_track_id = None
        min_dist = float("inf")

        for track_id, track in self.tracks.items():
            if frame_idx - track["last_frame"] > self.max_missed_frames:
                continue

            tcx, tcy = track["last_center"]
            dist = math.hypot(cx - tcx, cy - tcy)

            if dist < self.max_dist and dist < min_dist:
                min_dist = dist
                best_track_id = track_id

        return best_track_id

    def should_run_ocr(self, track_id: int, crop: np.ndarray, frame_idx: int) -> Tuple[bool, bool]:
        """
        Determine whether OCR is needed for this detection:
        Returns: (run_ocr: bool, use_multi_attempt: bool)
        """
        if track_id is None or track_id not in self.tracks:
            return True, True

        track = self.tracks[track_id]
        obs = track["observations"]

        if not obs:
            return True, True

        high_conf_obs = [o for o in obs if o[1] >= 0.88 and o[0]]

        # If we already have >= 2 solid readings, skip redundant OCR on every frame
        if len(high_conf_obs) >= 2:
            # Only run if it's been at least 30 frames since last OCR
            last_ocr_frame = track.get("last_ocr_frame", 0)
            if (frame_idx - last_ocr_frame) < 30:
                return False, False

        # If track has moderate confidence, run single fast pass
        best_conf = max((o[1] for o in obs), default=0.0)
        if best_conf >= 0.75:
            return True, False

        # Low confidence or uncertain: run multi-attempt
        return True, True

    def get_cached_prediction(self, track_id: int) -> Dict[str, Any]:
        """
        Retrieve best cached prediction for a track when skipping OCR.
        """
        if track_id not in self.tracks or not self.tracks[track_id]["observations"]:
            return {
                "plate_text": "",
                "cleaned_text": "",
                "ocr_confidence": 0.0,
                "valid_indian_plate": False,
                "validation_reason": "No OCR",
                "ocr_attempts": 0
            }

        track = self.tracks[track_id]
        best_obs = max(track["observations"], key=lambda x: x[1])
        return {
            "plate_text": best_obs[0],
            "cleaned_text": best_obs[0],
            "ocr_confidence": best_obs[1],
            "valid_indian_plate": best_obs[2],
            "validation_reason": "Reused from active track",
            "ocr_attempts": 0
        }

    def update_track(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        cleaned_text: str,
        ocr_conf: float,
        is_valid: bool,
        yolo_conf: float,
        frame_idx: int,
        plate_crop: np.ndarray = None,
        ran_ocr: bool = True
    ) -> int:
        """
        Update or create track with current observation.
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if track_id is None or track_id not in self.tracks:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = {
                "first_frame": frame_idx,
                "last_frame": frame_idx,
                "last_center": (cx, cy),
                "last_bbox": bbox,
                "last_ocr_frame": frame_idx if ran_ocr else 0,
                "observations": [],
                "best_crop": plate_crop,
                "printed": False
            }

        track = self.tracks[track_id]
        track["last_frame"] = frame_idx
        track["last_center"] = (cx, cy)
        track["last_bbox"] = bbox

        if ran_ocr:
            track["last_ocr_frame"] = frame_idx

        if cleaned_text:
            track["observations"].append((cleaned_text, ocr_conf, is_valid, yolo_conf, frame_idx))

        if plate_crop is not None and (track["best_crop"] is None or ocr_conf > max((obs[1] for obs in track["observations"][:-1]), default=0)):
            track["best_crop"] = plate_crop

        return track_id

    def get_resolved_tracks(self) -> List[Dict[str, Any]]:
        """
        Run multi-frame voting on all tracked vehicles.
        """
        results = []
        for track_id, track in self.tracks.items():
            obs = [(t, c, v) for t, c, v, y, f in track["observations"] if t]
            if not obs:
                results.append({
                    "track_id": track_id,
                    "consensus_plate": "NOT READ",
                    "consensus_conf": 0.0,
                    "is_valid": False,
                    "observations_count": len(track["observations"]),
                    "first_frame": track["first_frame"],
                    "last_frame": track["last_frame"]
                })
                continue

            consensus_plate, consensus_conf, is_valid = vote_plate_consensus(obs)
            results.append({
                "track_id": track_id,
                "consensus_plate": consensus_plate,
                "consensus_conf": consensus_conf,
                "is_valid": is_valid,
                "observations_count": len(track["observations"]),
                "first_frame": track["first_frame"],
                "last_frame": track["last_frame"]
            })

        return results


def query_vehicle_history(plate: str, db_path: str = "data/vehicle_history.db"):
    """
    Query and display chronological detections and trajectory waypoints for a given license plate.
    """
    store = VehicleHistoryStore(db_path)
    records = store.get_history_by_plate(plate)
    print("=" * 66, flush=True)
    print(f"        VEHICLE DETECTION HISTORY & TRAJECTORY QUERY", flush=True)
    print("=" * 66, flush=True)
    print(f"Query Plate Target : {plate.upper()}", flush=True)
    print(f"Database Location  : {db_path}", flush=True)
    print(f"Total Sightings    : {len(records)}", flush=True)
    print("-" * 66, flush=True)

    if not records:
        print(f"No recorded sightings found for plate '{plate}'.", flush=True)
        print("=" * 66, flush=True)
        return

    print(f"{'#':<3} | {'Timestamp':<23} | {'Camera ID':<16} | {'Frame':<6} | {'Track':<5} | {'GPS (Lat, Lon)':<18} | {'OCR Conf':<8} | {'Valid'}", flush=True)
    print("-" * 66, flush=True)
    for idx, r in enumerate(records, 1):
        gps_str = f"({r.latitude:.4f}, {r.longitude:.4f})"
        valid_str = "YES" if r.valid_indian_plate else "NO"
        print(f"{idx:<3} | {r.timestamp:<23} | {r.camera_id:<16} | {r.frame_number:<6} | #{r.track_id:<4} | {gps_str:<18} | {r.ocr_confidence:.2f}     | {valid_str}", flush=True)

    print("=" * 66, flush=True)
    print(f"Trajectory Waypoints ({len(records)} points):", flush=True)
    for idx, r in enumerate(records, 1):
        print(f"  Step {idx}: Camera '{r.camera_id}' @ {r.timestamp} -> Lat: {r.latitude:.5f}, Lon: {r.longitude:.5f} (Track #{r.track_id})", flush=True)
    print("=" * 66, flush=True)


def run_video_anpr(
    video_path: str = "data/videos/traffic.mp4",
    frame_skip: int = 15,
    output_crop_dir: str = "data/plates/video",
    conf_threshold: float = 0.32,
    imgsz: int = 1280,
    camera_id: str = "CAM_01_JUNCTION_A",
    latitude: float = 28.6139,
    longitude: float = 77.2090,
    db_path: str = "data/vehicle_history.db",
    max_frames: int = None
):
    """
    Run CPU-optimized ANPR pipeline on a video file with track-aware OCR,
    multi-frame voting, and persistent vehicle detection record storage.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at: {video_path}", flush=True)
        return

    os.makedirs(output_crop_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video: {video_path}", flush=True)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames and max_frames > 0:
        total_frames = min(total_frames, max_frames)

    print("=" * 56, flush=True)
    print("       VEHICLE MONITORING - ANPR VIDEO RUNNER", flush=True)
    print("=" * 56, flush=True)
    print(f"Video Source     : {video_path}", flush=True)
    print(f"Camera ID        : {camera_id} (GPS: {latitude:.4f}, {longitude:.4f})", flush=True)
    print(f"Database Path    : {db_path}", flush=True)
    print(f"Total Frames     : {total_frames}", flush=True)
    print(f"FPS              : {fps:.2f}", flush=True)
    print(f"Frame Sampling   : Every {frame_skip} frames ({fps / frame_skip:.1f} FPS processed)", flush=True)
    print(f"Output Crop Dir  : {output_crop_dir}", flush=True)
    print("=" * 56, flush=True)
    print("Initializing ANPR Pipeline & Vehicle History Store...\n", flush=True)

    pipeline = ANPRPipeline()
    tracker = VehiclePlateTracker(max_dist=250.0, max_missed_frames=frame_skip * 4)
    history_store = VehicleHistoryStore(db_path=db_path)

    base_time = datetime(2026, 9, 15, 10, 0, 0)
    start_time = time.time()

    processed_frames = 0
    total_plate_detections = 0
    total_ocr_attempts = 0
    valid_indian_plates_count = 0
    rejected_detections_count = 0
    reused_ocr_count = 0
    saved_records_count = 0

    seen_plates_best = {}
    last_printed_frame = {}

    frame_idx = 0

    while True:
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            processed_frames += 1
            sec_elapsed = frame_idx / fps
            frame_datetime = base_time + timedelta(seconds=sec_elapsed)
            timestamp_iso = frame_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

            # 1. Scaled YOLO Plate Detection
            raw_dets = pipeline.detector.detect(frame, conf=conf_threshold, imgsz=imgsz)

            for det_idx, det in enumerate(raw_dets):
                total_plate_detections += 1
                bbox = det["bbox"]
                yolo_conf = det["confidence"]
                x1, y1, x2, y2 = bbox
                crop = extract_plate_crop(frame, bbox, margin_ratio=0.015)

                # Match against active tracks
                track_id = tracker.match_bbox(bbox, frame_idx)
                run_ocr, use_multi = tracker.should_run_ocr(track_id, crop, frame_idx)

                if run_ocr:
                    if use_multi:
                        plate_text, ocr_conf, ocr_attempts = pipeline.ocr.recognize_with_variants(crop)
                    else:
                        plate_text, ocr_conf = pipeline.ocr.recognize(crop)
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
                    reason = cached["validation_reason"]
                    ocr_attempts = 0
                    reused_ocr_count += 1
                    ran_ocr = False

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

                # Persist Vehicle Detection Record
                record = VehicleDetectionRecord(
                    plate_text=plate_text,
                    cleaned_plate=cleaned_text or plate_text,
                    yolo_confidence=yolo_conf,
                    ocr_confidence=ocr_conf,
                    valid_indian_plate=is_valid,
                    camera_id=camera_id,
                    timestamp=timestamp_iso,
                    frame_number=frame_idx,
                    track_id=track_id,
                    latitude=latitude,
                    longitude=longitude
                )
                history_store.insert_record(record)
                saved_records_count += 1

                if is_valid:
                    valid_indian_plates_count += 1
                else:
                    rejected_detections_count += 1

                # Save crop
                crop_filename = f"frame_{frame_idx:05d}_plate_{det_idx + 1}_{cleaned_text or 'unknown'}.jpg"
                crop_filepath = os.path.join(output_crop_dir, crop_filename)
                if crop is not None and crop.size > 0:
                    cv2.imwrite(crop_filepath, crop)

                # Output logging
                should_print = False
                key = cleaned_text or f"unread_{bbox}"

                if cleaned_text:
                    if cleaned_text not in seen_plates_best:
                        seen_plates_best[cleaned_text] = ocr_conf
                        should_print = True
                    else:
                        prev_conf = seen_plates_best[cleaned_text]
                        prev_frame = last_printed_frame.get(cleaned_text, -999)
                        if ocr_conf > (prev_conf + 0.08) or (frame_idx - prev_frame > 30):
                            if ocr_conf > prev_conf:
                                seen_plates_best[cleaned_text] = ocr_conf
                            should_print = True
                else:
                    last_unread = last_printed_frame.get("__unread__", -999)
                    if (frame_idx - last_unread) > (frame_skip * 3):
                        should_print = True
                        key = "__unread__"

                if should_print:
                    last_printed_frame[key] = frame_idx
                    print(f"Frame: {frame_idx} | Track: #{track_id:02d}", flush=True)
                    print(f"Timestamp: {sec_elapsed:.2f} sec ({timestamp_iso})", flush=True)
                    print(f"Plate: {cleaned_text or plate_text or 'NOT READ'}", flush=True)
                    print(f"YOLO: {yolo_conf:.2f} | OCR: {ocr_conf:.2f}", flush=True)
                    print(f"Indian Plate: {'VALID' if is_valid else 'INVALID'} ({reason})", flush=True)
                    print(flush=True)

        frame_idx += 1

    cap.release()
    total_time = time.time() - start_time

    # Export history JSON
    json_path = "data/vehicle_history.json"
    history_store.export_to_json(json_path)

    resolved_tracks = tracker.get_resolved_tracks()
    unique_valid_plates = [t for t in resolved_tracks if t["is_valid"]]
    unique_all_plates = [t for t in resolved_tracks if t["consensus_plate"] != "NOT READ"]
    db_stats = history_store.get_statistics()

    # Final Summary Evaluation Report
    print("=" * 56, flush=True)
    print("             ANPR & HISTORY EVALUATION REPORT", flush=True)
    print("=" * 56, flush=True)
    print(f"Total Frames in Video        : {total_frames}", flush=True)
    print(f"Processed Frames             : {processed_frames} ({fps / frame_skip:.1f} FPS sampling)", flush=True)
    print(f"Total Plate Detections       : {total_plate_detections}", flush=True)
    print(f"Total OCR Inference Attempts : {total_ocr_attempts}", flush=True)
    print(f"Reused OCR Predictions       : {reused_ocr_count}", flush=True)
    print(f"Raw Valid Indian Detections  : {valid_indian_plates_count}", flush=True)
    print(f"Rejected Non-Indian Plates   : {rejected_detections_count}", flush=True)
    print(f"Total Tracked Vehicles       : {len(resolved_tracks)}", flush=True)
    print(f"Final Unique Plates (Voted)  : {len(unique_all_plates)}", flush=True)
    print(f"Final Valid Indian Plates    : {len(unique_valid_plates)}", flush=True)
    print(f"Detection Records Stored     : {db_stats['total_detections']} (SQLite: {db_path})", flush=True)
    print(f"Exported History JSON        : {json_path}", flush=True)
    print(f"Processing Time              : {total_time:.2f} sec ({processed_frames / total_time if total_time > 0 else 0:.1f} FPS)", flush=True)
    print("-" * 56, flush=True)
    print("Multi-Frame Voting Consensus Summary:", flush=True)
    for t in resolved_tracks:
        status = "VALID (Indian)" if t["is_valid"] else "FOREIGN / INVALID"
        print(f"  Track #{t['track_id']:02d}: {t['consensus_plate']:12} | Conf: {t['consensus_conf']:.2f} | Frames: {t['observations_count']} | Status: {status}", flush=True)

    print("-" * 56, flush=True)
    print("Target 90% Accuracy Assessment:", flush=True)
    if len(unique_valid_plates) == 0:
        print(">> Note: 'data/videos/traffic.mp4' contains UK/European format registration plates (e.g., HW51VSU, NA13NRU, GX15OGJ, KH05ZZK, WR02FKD).", flush=True)
        print(">> 90% accuracy cannot yet be verified from this video.", flush=True)
    else:
        print(f">> Detected {len(unique_valid_plates)} valid Indian plates.", flush=True)
    print("=" * 56, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VehicleMonitoring ANPR Video Processor & History Store")
    parser.add_argument("--video", type=str, default="data/videos/traffic.mp4", help="Path to input video")
    parser.add_argument("--frame-skip", type=int, default=15, help="Frame skipping interval (default: 15)")
    parser.add_argument("--conf", type=float, default=0.32, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO input image size")
    parser.add_argument("--out-dir", type=str, default="data/plates/video", help="Output directory for plate crops")
    parser.add_argument("--camera-id", type=str, default="CAM_01_JUNCTION_A", help="Camera identifier")
    parser.add_argument("--lat", type=float, default=28.6139, help="Camera latitude")
    parser.add_argument("--lon", type=float, default=77.2090, help="Camera longitude")
    parser.add_argument("--db-path", type=str, default="data/vehicle_history.db", help="SQLite database path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process for fast testing")
    parser.add_argument("--query-plate", type=str, default=None, help="Query chronological detection history for a plate")
    args = parser.parse_args()

    if args.query_plate:
        query_vehicle_history(args.query_plate, db_path=args.db_path)
    else:
        run_video_anpr(
            video_path=args.video,
            frame_skip=args.frame_skip,
            output_crop_dir=args.out_dir,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            camera_id=args.camera_id,
            latitude=args.lat,
            longitude=args.lon,
            db_path=args.db_path,
            max_frames=args.max_frames
        )
