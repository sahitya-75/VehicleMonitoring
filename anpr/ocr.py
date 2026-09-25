import os
import sys
import logging
from typing import Tuple, List
import numpy as np

# Suppress Paddle and PaddleOCR verbose logs
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddlex").setLevel(logging.ERROR)

from paddleocr import PaddleOCR
from anpr.preprocessing import (
    enhance_clahe,
    enhance_sharpen,
    enhance_denoise,
    enhance_threshold,
    is_valid_crop
)
from anpr.plate_validator import validate_indian_plate


class PlateOCR:
    """
    PaddleOCR character recognizer for vehicle license plates.
    Loads PaddleOCR once on CPU and provides fast, accurate recognition with multi-attempt fallbacks.
    """

    _instance = None

    def __init__(self, lang: str = "en"):
        # CPU-optimized PaddleOCR instance
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

    @classmethod
    def get_instance(cls, lang: str = "en"):
        """Singleton helper to avoid multiple heavy model initializations."""
        if cls._instance is None:
            cls._instance = cls(lang=lang)
        return cls._instance

    def recognize_single(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Run a single OCR prediction pass on an image.

        Args:
            plate_image: np.ndarray (BGR or preprocessed crop).

        Returns:
            Tuple: (plate_text: str, ocr_confidence: float)
        """
        if plate_image is None or not isinstance(plate_image, np.ndarray) or plate_image.size == 0:
            return "", 0.0

        try:
            results = self.ocr.predict(plate_image)
            if not results:
                return "", 0.0

            recognized_texts = []
            recognized_scores = []

            for item in results:
                data = item.json if hasattr(item, "json") else item
                if isinstance(data, dict):
                    res = data.get("res", data)
                    texts = res.get("rec_texts", [])
                    scores = res.get("rec_scores", [])

                    for t, s in zip(texts, scores):
                        cleaned_str = str(t).strip()
                        if cleaned_str:
                            recognized_texts.append(cleaned_str)
                            recognized_scores.append(float(s))

            if not recognized_texts:
                return "", 0.0

            combined_text = " ".join(recognized_texts)
            avg_score = float(np.mean(recognized_scores)) if recognized_scores else 0.0

            return combined_text, round(avg_score, 4)

        except Exception:
            return "", 0.0

    def recognize(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text with default primary CLAHE preprocessing.
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        is_val, _ = is_valid_crop(plate_crop)
        if not is_val:
            return "", 0.0
        preprocessed = enhance_clahe(plate_crop)
        return self.recognize_single(preprocessed)

    def recognize_with_variants(
        self,
        plate_crop: np.ndarray,
        fast_accept_conf: float = 0.78
    ) -> Tuple[str, float, int]:
        """
        Multi-attempt OCR with crop quality gating and format-aware variant selection:
        1. Filters invalid/degenerate crops before running OCR.
        2. Fast-accepts when primary CLAHE yields strong confidence & valid format.
        3. Evaluates Sharpen, Denoise, or Threshold only when needed.
        4. Selects the optimal candidate balancing OCR confidence and Indian plate validity.

        Args:
            plate_crop: Raw BGR plate crop.
            fast_accept_conf: Threshold to accept immediately without extra passes.

        Returns:
            Tuple: (best_text: str, best_confidence: float, total_attempts: int)
        """
        if plate_crop is None or not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
            return "", 0.0, 0

        # Step 0: Quality pre-check (avoids expensive OCR on noise artifacts)
        is_val_crop, _ = is_valid_crop(plate_crop)
        if not is_val_crop:
            return "", 0.0, 0

        # Attempt 1: Standard CLAHE enhancement (fast path)
        text_clahe, conf_clahe = self.recognize_single(enhance_clahe(plate_crop))
        is_val_clahe, clean_clahe, _ = validate_indian_plate(text_clahe)
        attempts = 1

        # Fast accept: very high confidence + valid Indian plate format
        if text_clahe and is_val_clahe and conf_clahe >= 0.92:
            return text_clahe, conf_clahe, attempts

        best_text = text_clahe
        best_conf = conf_clahe
        best_is_valid = is_val_clahe

        # Attempt 2: Sharpened variant (unsharp masking to resolve character edge ambiguities)
        text_sharp, conf_sharp = self.recognize_single(enhance_sharpen(plate_crop))
        is_val_sharp, clean_sharp, _ = validate_indian_plate(text_sharp)
        attempts += 1

        # Format-weighted candidate selection:
        # 1. Prefer valid Indian plate over invalid
        # 2. When both valid, prefer cleaner registration structure or higher confidence
        sharp_score = conf_sharp * (1.3 if is_val_sharp else 1.0)
        best_score = best_conf * (1.3 if best_is_valid else 1.0)

        # If sharpen resolves into a valid plate while clahe was invalid/empty
        if is_val_sharp and not best_is_valid:
            best_text = text_sharp
            best_conf = conf_sharp
            best_is_valid = True
            best_score = sharp_score
        elif sharp_score > best_score or (not best_text and text_sharp):
            best_text = text_sharp
            best_conf = conf_sharp
            best_is_valid = is_val_sharp
            best_score = sharp_score

        if best_text and best_is_valid and best_conf >= fast_accept_conf:
            return best_text, best_conf, attempts

        # Attempt 3: Denoised variant (if still low confidence or unreadable)
        if best_conf < 0.70 or not best_text or not best_is_valid:
            text_denoise, conf_denoise = self.recognize_single(enhance_denoise(plate_crop))
            is_val_denoise, clean_denoise, _ = validate_indian_plate(text_denoise)
            attempts += 1

            denoise_score = conf_denoise * (1.3 if is_val_denoise else 1.0)
            if (is_val_denoise and not best_is_valid) or denoise_score > best_score or (not best_text and text_denoise):
                best_text = text_denoise
                best_conf = conf_denoise
                best_is_valid = is_val_denoise
                best_score = denoise_score

        # Attempt 4: Threshold variant (only if completely empty after 3 attempts)
        if not best_text:
            text_thresh, conf_thresh = self.recognize_single(enhance_threshold(plate_crop))
            attempts += 1
            if text_thresh:
                best_text = text_thresh
                best_conf = conf_thresh

        return best_text, best_conf, attempts


# Standalone module test
if __name__ == "__main__":
    import cv2
    ocr_engine = PlateOCR.get_instance()
    test_img = cv2.imread("data/images/indian_car.jpg")
    if test_img is not None:
        crop = test_img[396:414, 457:513]
        text, conf, atts = ocr_engine.recognize_with_variants(crop)
        print(f"Multi-attempt OCR Result: '{text}', Confidence: {conf:.2f}, Attempts: {atts}")
    else:
        print("Test image not found.")
