from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from anpr.detector import PlateDetector
from anpr.preprocessing import preprocess_plate
from anpr.ocr import PlateOCR
from anpr.plate_validator import validate_indian_plate


def extract_plate_crop(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_ratio: float = 0.015
) -> np.ndarray:
    """
    Extract license plate crop from image with an adaptive border margin
    to eliminate frame borders, screws, and vehicle body edge artifacts.

    Args:
        image: OpenCV BGR image (np.ndarray).
        bbox: Bounding box tuple (x1, y1, x2, y2).
        margin_ratio: Percentage of width/height to trim from borders (default: 0.015 / 1.5%).

    Returns:
        np.ndarray: Clean cropped license plate image.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    # Inward margin: trim outer border/bezel artifact while keeping characters intact
    dx_r = max(1, int(round(w * margin_ratio))) if margin_ratio > 0 else 0
    dy = int(round(h * margin_ratio)) if margin_ratio > 0 else 0

    x1_c = max(0, min(x1, x2 - 1))
    y1_c = max(0, min(y1 + dy, y2 - 1))
    x2_c = max(x1_c + 1, min(x2 - dx_r, image.shape[1]))
    y2_c = max(y1_c + 1, min(y2 - dy, image.shape[0]))

    return image[y1_c:y2_c, x1_c:x2_c].copy()


class ANPRPipeline:
    """
    Reusable End-to-End Automatic Number Plate Recognition (ANPR) Pipeline.
    Integrates YOLO Detection, Preprocessing, Multi-attempt PaddleOCR Recognition, and Indian Plate Validation.
    """

    def __init__(self, detector: Optional[PlateDetector] = None, ocr: Optional[PlateOCR] = None):
        self.detector = detector if detector is not None else PlateDetector()
        self.ocr = ocr if ocr is not None else PlateOCR.get_instance()

    def process_frame(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        imgsz: int = 1280,
        multi_attempt: bool = True,
        crop_margin: float = 0.02
    ) -> List[Dict[str, Any]]:
        """
        Process a single image or video frame through the full ANPR pipeline.

        Args:
            frame: OpenCV BGR image (np.ndarray).
            conf: YOLO detection confidence threshold.
            imgsz: Resolution for YOLO detection.
            multi_attempt: Whether to evaluate preprocessing variants on low OCR confidence.
            crop_margin: Inward border margin ratio to trim plate edges.

        Returns:
            List of detection dicts:
            [
                {
                    "plate_text": str,
                    "cleaned_text": str,
                    "yolo_confidence": float,
                    "ocr_confidence": float,
                    "ocr_attempts": int,
                    "valid_indian_plate": bool,
                    "validation_reason": str,
                    "bbox": (x1, y1, x2, y2),
                    "plate_crop": np.ndarray
                },
                ...
            ]
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        # 1. YOLO Plate Detection
        detections = self.detector.detect(frame, conf=conf, imgsz=imgsz)
        results = []

        for det in detections:
            bbox = det["bbox"]
            yolo_conf = det["confidence"]

            # 2. Extract clean plate crop with inward margin
            plate_crop = extract_plate_crop(frame, bbox, margin_ratio=crop_margin)
            if plate_crop is None or plate_crop.size == 0:
                continue

            # 3. Multi-attempt OCR with Preprocessing Variants
            if multi_attempt:
                plate_text, ocr_conf, ocr_attempts = self.ocr.recognize_with_variants(plate_crop)
            else:
                plate_text, ocr_conf = self.ocr.recognize(plate_crop)
                ocr_attempts = 1

            # 4. Indian Plate Validation
            is_valid, cleaned_text, reason = validate_indian_plate(plate_text)

            results.append({
                "plate_text": plate_text,
                "cleaned_text": cleaned_text,
                "yolo_confidence": yolo_conf,
                "ocr_confidence": ocr_conf,
                "ocr_attempts": ocr_attempts,
                "valid_indian_plate": is_valid,
                "validation_reason": reason,
                "bbox": bbox,
                "plate_crop": plate_crop
            })

        return results


# Standalone module test
if __name__ == "__main__":
    import cv2
    import os

    pipeline = ANPRPipeline()
    test_path = "data/images/indian_car.jpg"
    if os.path.exists(test_path):
        img = cv2.imread(test_path)
        detections = pipeline.process_frame(img)
        print(f"\nPipeline Detections ({len(detections)}):")
        for i, d in enumerate(detections, 1):
            print(f"[{i}] Text: '{d['plate_text']}' | Cleaned: '{d['cleaned_text']}' | YOLO: {d['yolo_confidence']:.2f} | OCR: {d['ocr_confidence']:.2f} (Attempts: {d['ocr_attempts']}) | Valid: {d['valid_indian_plate']} ({d['validation_reason']})")
    else:
        print("Test image not found.")
