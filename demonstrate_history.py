import os
import sys
import json
import time
from datetime import datetime, timedelta

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from anpr.pipeline import ANPRPipeline
from anpr.storage import VehicleDetectionRecord, VehicleHistoryStore
from video_anpr import run_video_anpr, query_vehicle_history
import cv2


def demonstrate_end_to_end_history():
    """
    Demonstrate complete end-to-end flow:
    ANPR -> Vehicle Tracking -> Plate Recognition -> Timestamp -> Camera ID -> Detection Record -> Saved Vehicle History & Trajectory Reconstruction.
    """
    db_path = "data/vehicle_history.db"
    store = VehicleHistoryStore(db_path=db_path)
    store.clear()  # Start with fresh demo database

    print("=" * 70, flush=True)
    print("      SIH 2026 - VEHICLE DETECTION HISTORY & TRAJECTORY SYSTEM", flush=True)
    print("=" * 70, flush=True)
    print(f"Database Storage : SQLite ({db_path})", flush=True)
    print(f"Record Model     : 10 attributes (plate, confidences, camera, GPS, timestamp, track)", flush=True)
    print("=" * 70, flush=True)

    # -------------------------------------------------------------
    # Step 1: Run ANPR Video Processing on traffic.mp4
    # -------------------------------------------------------------
    video_path = "data/videos/traffic.mp4"
    if os.path.exists(video_path):
        print("\n>>> STEP 1: Processing Video Traffic Stream through ANPR & Tracker...", flush=True)
        # Process a sample chunk (e.g. 150 frames = 10 sampled frames) for fast demonstration
        run_video_anpr(
            video_path=video_path,
            frame_skip=15,
            conf_threshold=0.32,
            camera_id="CAM_01_JUNCTION_A",
            latitude=28.6139,
            longitude=77.2090,
            db_path=db_path,
            max_frames=150
        )
    else:
        print(f"Warning: Video not found at {video_path}", flush=True)

    # -------------------------------------------------------------
    # Step 2: Multi-Camera Network Simulation for Indian Vehicles
    # (Ingesting real detected Indian plates across city junctions)
    # -------------------------------------------------------------
    print("\n>>> STEP 2: Ingesting Multi-Camera City Network Detections for Indian Plates...", flush=True)
    pipeline = ANPRPipeline()
    img = cv2.imread("data/images/indian_car.jpg")

    if img is not None:
        # Run real ANPR on indian_car.jpg
        dets = pipeline.process_frame(img, crop_margin=0.015)
        print(f"Detected {len(dets)} vehicles in Indian scene image:", flush=True)

        # Multi-camera network simulation representing vehicle movements across Delhi-NCR junctions
        camera_network = [
            {"camera_id": "CAM_01_DELHI_NORTH", "lat": 28.7041, "lon": 77.1025, "time_offset_min": 0},
            {"camera_id": "CAM_02_DELHI_RING_ROAD", "lat": 28.6353, "lon": 77.2250, "time_offset_min": 15},
            {"camera_id": "CAM_03_DELHI_SOUTH", "lat": 28.5355, "lon": 77.2410, "time_offset_min": 35},
            {"camera_id": "CAM_04_NOIDA_EXPRESSWAY", "lat": 28.5020, "lon": 77.3820, "time_offset_min": 60}
        ]

        base_time = datetime(2026, 9, 15, 9, 0, 0)
        track_counter = 100

        for cam_idx, cam in enumerate(camera_network):
            for d in dets:
                plate = d["cleaned_text"]
                if not plate:
                    continue

                # Simulated sighting time
                sighting_time = base_time + timedelta(minutes=cam["time_offset_min"], seconds=cam_idx*12)
                timestamp_iso = sighting_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                track_counter += 1

                record = VehicleDetectionRecord(
                    plate_text=d["plate_text"],
                    cleaned_plate=plate,
                    yolo_confidence=d["yolo_confidence"],
                    ocr_confidence=d["ocr_confidence"],
                    valid_indian_plate=d["valid_indian_plate"],
                    camera_id=cam["camera_id"],
                    timestamp=timestamp_iso,
                    frame_number=cam_idx * 150 + 15,
                    track_id=track_counter,
                    latitude=cam["lat"],
                    longitude=cam["lon"]
                )
                store.insert_record(record)
                print(f"  [+] Logged '{plate:10}' @ {cam['camera_id']:22} (GPS: {cam['lat']:.4f}, {cam['lon']:.4f}) | {timestamp_iso}", flush=True)

    # -------------------------------------------------------------
    # Step 3: Export & Query Chronological Vehicle History
    # -------------------------------------------------------------
    print("\n>>> STEP 3: Querying Chronological History & Trajectory...", flush=True)
    json_path = store.export_to_json("data/vehicle_history.json")
    print(f"Exported JSON snapshot to: {json_path}", flush=True)

    test_queries = ["HR51BC3493", "UP21X3666", "DL3CC8387"]
    for q_plate in test_queries:
        print()
        query_vehicle_history(q_plate, db_path=db_path)

    # Database Summary Statistics
    stats = store.get_statistics()
    print("\n" + "=" * 70, flush=True)
    print("                 FINAL STORAGE & ANPR SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Total Detection Records Saved : {stats['total_detections']}", flush=True)
    print(f"Unique Tracked Plates         : {stats['unique_plates']}", flush=True)
    print(f"Unique Vehicle Track IDs      : {stats['unique_tracks']}", flush=True)
    print(f"Valid Indian Plate Records    : {stats['valid_indian_plates']}", flush=True)
    print(f"SQLite Database File          : {stats['db_path']}", flush=True)
    print(f"JSON Export File              : data/vehicle_history.json", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    demonstrate_end_to_end_history()
