import os
import sys
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from anpr.pipeline import ANPRPipeline


def test_image_anpr(image_path: str = "data/images/indian_car.jpg"):
    if not os.path.exists(image_path):
        # Fallback to any available image in data/images
        fallback_dir = "data/images"
        if os.path.exists(fallback_dir):
            images = [os.path.join(fallback_dir, f) for f in os.listdir(fallback_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if images:
                image_path = images[0]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found at {image_path}!")
        return

    pipeline = ANPRPipeline()
    detections = pipeline.process_frame(image)

    os.makedirs("data/plates", exist_ok=True)

    print("\n======================================")
    print("        ANPR VEHICLE MONITORING")
    print("======================================")
    print(f"Image: {image_path}")
    print(f"Total plates detected: {len(detections)}")

    for i, det in enumerate(detections, 1):
        plate_text = det["plate_text"]
        cleaned_text = det["cleaned_text"]
        yolo_conf = det["yolo_confidence"]
        ocr_conf = det["ocr_confidence"]
        valid = det["valid_indian_plate"]
        reason = det["validation_reason"]

        crop = det.get("plate_crop")
        if crop is not None and crop.size > 0:
            crop_path = f"data/plates/plate_{i}.jpg"
            cv2.imwrite(crop_path, crop)

        print("\n--------------------------------------")
        print(f"Plate {i}")
        print(f"OCR Text        : {plate_text or 'NOT READ'}")
        print(f"Cleaned Text    : {cleaned_text or 'N/A'}")
        print(f"YOLO Confidence : {yolo_conf:.2f}")
        print(f"OCR Confidence  : {ocr_conf:.2f}")
        print(f"Indian Format   : {'VALID' if valid else 'INVALID'}")
        print(f"Reason          : {reason}")

    print("\n======================================")
    print("              COMPLETED")
    print("======================================")


if __name__ == "__main__":
    test_image_anpr()