from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.types import PerceptionFrame
from src.perception.video.ocr import TesseractOCRBackend, VideoTextExtractor


def make_text_frame(text: str) -> np.ndarray:
    image = np.full((240, 900, 3), 255, dtype=np.uint8)
    if text:
        cv2.putText(
            image,
            text,
            (35, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.4,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
    return image


def test_real_tesseract_ocr_and_metadata() -> None:
    backend = TesseractOCRBackend()
    extractor = VideoTextExtractor(backend)
    frame = PerceptionFrame(
        image=make_text_frame("FRIDAY VISIONAID 204"),
        timestamp=4.2,
        source_id="synthetic:video-ocr-smoke",
        frame_index=126,
    )

    observations = extractor.process(frame)
    recognized_text = " ".join(item.text for item in observations)
    normalized_text = recognized_text.upper()
    assert "FRIDAY" in normalized_text, recognized_text
    assert "VISIONAID" in normalized_text, recognized_text
    assert "204" in normalized_text, recognized_text
    assert all(item.timestamp == 4.2 for item in observations)
    assert all(item.frame_index == 126 for item in observations)
    assert all(item.source_id == frame.source_id for item in observations)
    assert all(item.bbox is not None for item in observations)
    assert all(item.confidence is not None for item in observations)
    assert all(0.0 <= item.confidence <= 1.0 for item in observations)
    assert all(
        item.bbox[2] > item.bbox[0] and item.bbox[3] > item.bbox[1]
        for item in observations
        if item.bbox is not None
    )
    print(f"REAL OCR RECOGNIZED: {recognized_text}")
    print("TEXT, CONFIDENCE, BOUNDING BOX, AND FRAME METADATA: PASS")


def test_empty_real_ocr_result() -> None:
    backend = TesseractOCRBackend()
    observations = VideoTextExtractor(backend).process(
        PerceptionFrame(
            image=make_text_frame(""),
            timestamp=0.0,
            source_id="synthetic:blank-ocr-smoke",
            frame_index=0,
        )
    )
    assert observations == ()
    print("EMPTY IMAGE RETURNS NO TEXT OBSERVATIONS: PASS")


def main() -> None:
    test_real_tesseract_ocr_and_metadata()
    test_empty_real_ocr_result()
    print("REAL VIDEO OCR SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
