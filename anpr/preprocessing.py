import cv2
import numpy as np
from typing import List, Tuple


def resize_plate(plate_crop: np.ndarray, target_height: int = 80) -> np.ndarray:
    """
    Upscale plate crop to target height using bicubic interpolation.
    """
    if plate_crop is None or not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
        return plate_crop

    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return plate_crop

    if h < target_height:
        scale = target_height / float(h)
        new_w = max(1, int(w * scale))
        return cv2.resize(plate_crop, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
    elif w < 160:
        scale = 160.0 / float(w)
        new_h = max(1, int(h * scale))
        return cv2.resize(plate_crop, (160, new_h), interpolation=cv2.INTER_CUBIC)

    return plate_crop.copy()


def enhance_clahe(plate_crop: np.ndarray) -> np.ndarray:
    """
    Upscale + Grayscale + CLAHE contrast enhancement.
    """
    resized = resize_plate(plate_crop)
    if resized is None or resized.size == 0:
        return plate_crop

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def enhance_sharpen(plate_crop: np.ndarray) -> np.ndarray:
    """
    Upscale + CLAHE + Unsharp Masking filter for crisp character edges.
    """
    enhanced = enhance_clahe(plate_crop)
    if enhanced is None or enhanced.size == 0:
        return plate_crop

    # Unsharp mask
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened


def enhance_denoise(plate_crop: np.ndarray) -> np.ndarray:
    """
    Upscale + Bilateral Filter (edge-preserving denoise) + CLAHE.
    """
    resized = resize_plate(plate_crop)
    if resized is None or resized.size == 0:
        return plate_crop

    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=50, sigmaSpace=50)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY) if len(denoised.shape) == 3 else denoised
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def enhance_threshold(plate_crop: np.ndarray) -> np.ndarray:
    """
    Upscale + Grayscale + Otsu Binarization for difficult high-glare plates.
    """
    resized = resize_plate(plate_crop)
    if resized is None or resized.size == 0:
        return plate_crop

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def is_valid_crop(plate_crop: np.ndarray) -> Tuple[bool, str]:
    """
    Check if a cropped bounding box meets minimum quality standards for OCR.
    Filters out noise boxes, flat reflections, extreme aspect ratios, and severe blur.
    """
    if plate_crop is None or not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
        return False, "Empty crop"

    h, w = plate_crop.shape[:2]
    if h < 10 or w < 24:
        return False, f"Too small ({w}x{h})"

    aspect_ratio = w / float(h)
    if aspect_ratio < 1.2 or aspect_ratio > 8.0:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f})"

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 8.0:
        return False, f"Too blurry/flat (var={lap_var:.1f})"

    return True, "Valid quality"


def get_preprocessing_variants(plate_crop: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Return multiple preprocessing variants for fallback/multi-attempt OCR:
    1. CLAHE (standard contrast enhancement)
    2. Sharpen (unsharp mask for blurry characters)
    3. Denoise (bilateral filter + CLAHE for noisy crops)
    4. Threshold (Otsu binarization for high-glare/washed-out crops)
    """
    if plate_crop is None or not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
        return []

    return [
        ("clahe", enhance_clahe(plate_crop)),
        ("sharpen", enhance_sharpen(plate_crop)),
        ("denoise", enhance_denoise(plate_crop)),
        ("threshold", enhance_threshold(plate_crop)),
    ]


def preprocess_plate(plate_crop: np.ndarray, target_height: int = 80) -> np.ndarray:
    """
    Default primary preprocessor for OCR.
    """
    return enhance_clahe(plate_crop)


# Standalone module test
if __name__ == "__main__":
    test_crop = np.zeros((30, 90, 3), dtype=np.uint8)
    cv2.putText(test_crop, "AP05AB1234", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    is_val, reason = is_valid_crop(test_crop)
    print(f"Crop validity: {is_val} ({reason})")
    variants = get_preprocessing_variants(test_crop)
    print(f"Generated {len(variants)} preprocessing variants successfully.")
    for name, img in variants:
        print(f"  - Variant '{name}': shape={img.shape}")

