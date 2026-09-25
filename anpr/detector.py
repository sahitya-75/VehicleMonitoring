import os
import cv2
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    """
    License Plate Detector using YOLO.
    Loads models/license_plate.pt and detects bounding boxes and confidences.
    """

    def __init__(self, model_path: str = "models/license_plate.pt", device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray, conf: float = 0.25, imgsz: int = 1280, max_infer_dim: int = 1280):
        """
        Detect license plates in an image/frame.
        Automatically scales ultra-high-resolution (e.g. 4K) frames for fast CPU inference
        while returning bounding box coordinates in the original image coordinate space.

        Args:
            image: OpenCV BGR image (np.ndarray) or file path.
            conf: Confidence threshold for YOLO detections.
            imgsz: Input resolution for inference.
            max_infer_dim: Maximum dimension for CPU inference scaling.

        Returns:
            List of dicts:
            [
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float
                },
                ...
            ]
        """
        if image is None:
            return []

        if isinstance(image, np.ndarray) and image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]
        scale = 1.0
        infer_img = image

        # For large images (e.g. 4K 3840x2160), downscale for fast YOLO inference on CPU
        if max(orig_h, orig_w) > max_infer_dim:
            scale = max_infer_dim / float(max(orig_h, orig_w))
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            infer_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        results = self.model(
            infer_img,
            conf=conf,
            imgsz=min(imgsz, max_infer_dim),
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            bx1, by1, bx2, by2 = map(float, box.xyxy[0].tolist())

            # Rescale back to original image coordinates
            x1 = int(round(bx1 / scale))
            y1 = int(round(by1 / scale))
            x2 = int(round(bx2 / scale))
            y2 = int(round(by2 / scale))

            # Boundary clipping
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            bw = x2 - x1
            bh = y2 - y1

            # Filter degenerate or extremely small boxes (< 24px wide or < 10px high)
            if bw < 24 or bh < 10:
                continue

            # Filter extreme non-plate aspect ratios (< 1.2 or > 8.0)
            aspect = bw / float(bh)
            if aspect < 1.2 or aspect > 8.0:
                continue

            yolo_conf = float(box.conf[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": round(yolo_conf, 4)
            })

        return detections


# Standalone module test
if __name__ == "__main__":
    detector = PlateDetector()
    test_image_path = "data/images/indian_car.jpg"
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        dets = detector.detect(img)
        print(f"Detected {len(dets)} plates:")
        for i, d in enumerate(dets, 1):
            print(f"  {i}. bbox={d['bbox']}, confidence={d['confidence']:.2f}")
    else:
        print(f"Sample test image not found at: {test_image_path}")
