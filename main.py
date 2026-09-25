import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from anpr.pipeline import ANPRPipeline
from anpr.storage import VehicleDetectionRecord, VehicleHistoryStore
from video_anpr import run_video_anpr, query_vehicle_history
from anpr_test import test_image_anpr


def main():
    parser = argparse.ArgumentParser(description="VehicleMonitoring - ANPR & Vehicle Trajectory Tracking System")
    parser.add_argument("--mode", choices=["video", "image", "query", "list"], default="video", help="Mode: video, image, query, list")
    parser.add_argument("--input", type=str, default=None, help="Path to input video or image file")
    parser.add_argument("--frame-skip", type=int, default=15, help="Frame skipping interval for video (default: 15)")
    parser.add_argument("--query-plate", type=str, default=None, help="License plate number to look up historical sightings")
    parser.add_argument("--camera-id", type=str, default="CAM_01_JUNCTION_A", help="Camera ID")
    parser.add_argument("--lat", type=float, default=28.6139, help="Camera latitude")
    parser.add_argument("--lon", type=float, default=77.2090, help="Camera longitude")
    parser.add_argument("--db-path", type=str, default="data/vehicle_history.db", help="SQLite database path")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum video frames to process")
    args = parser.parse_args()

    if args.query_plate or args.mode == "query":
        target_plate = args.query_plate or "HR51BC3493"
        query_vehicle_history(target_plate, db_path=args.db_path)
    elif args.mode == "list":
        store = VehicleHistoryStore(args.db_path)
        unique = store.get_all_unique_plates()
        print("=" * 66)
        print(f"       RECORDED UNIQUE VEHICLES ({len(unique)} PLATES)")
        print("=" * 66)
        print(f"{'Plate':<14} | {'Sightings':<10} | {'Cameras':<8} | {'Max OCR':<8} | {'First Seen':<22} | {'Last Seen'}")
        print("-" * 66)
        for u in unique:
            print(f"{u['cleaned_plate']:<14} | {u['total_sightings']:<10} | {u['cameras_count']:<8} | {u['max_ocr_conf']:.2f}     | {u['first_seen']:<22} | {u['last_seen']}")
        print("=" * 66)
    elif args.mode == "video":
        video_path = args.input or "data/videos/traffic.mp4"
        run_video_anpr(
            video_path=video_path,
            frame_skip=args.frame_skip,
            camera_id=args.camera_id,
            latitude=args.lat,
            longitude=args.lon,
            db_path=args.db_path,
            max_frames=args.max_frames
        )
    elif args.mode == "image":
        image_path = args.input or "data/images/indian_car.jpg"
        test_image_anpr(image_path=image_path)


if __name__ == "__main__":
    main()