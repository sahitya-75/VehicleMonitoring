import os
import sys
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from anpr.detector import PlateDetector


def test_plate_detection(image_path: str = "data/images/indian_car.jpg"):
    if not os.path.exists(image_path):
        fallback_dir = "data/images"
        if os.path.exists(fallback_dir):
            images = [os.path.join(fallback_dir, f) for f in os.listdir(fallback_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if images:
                image_path = images[0]

    if not os.path.exists(image_path):
        print("No test images found in data/images")
        return

    image = cv2.imread(image_path)
    detector = PlateDetector()
    detections = detector.detect(image, conf=0.15, imgsz=1280)

    print(f"Image: {image_path}")
    print(f"License plates detected: {len(detections)}")

    for i, det in enumerate(detections, 1):
        bbox = det["bbox"]
        conf = det["confidence"]
        print(f"Plate {i}: bbox={bbox}, confidence={conf:.2f}")


if __name__ == "__main__":
    test_plate_detection()