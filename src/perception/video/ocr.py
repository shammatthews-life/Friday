from __future__ import annotations

import csv
import io
from collections import OrderedDict, deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, Protocol

import cv2
import numpy as np

from src.perception.types import PerceptionFrame


@dataclass(frozen=True)
class OCRDetection:
    """Text and optional metadata returned directly by an OCR backend."""

    text: str
    confidence: float | None = None
    bbox: tuple[float, float, float, float] | None = None


class OCRBackend(Protocol):
    """Backend contract: inspect an image and return only observed OCR metadata."""

    def extract(self, image: np.ndarray) -> Iterable[OCRDetection]:
        ...


class OCRBackendError(RuntimeError):
    """Raised by an OCR backend when it cannot extract text from an image."""


class TesseractOCRBackend:
    """Run the local Tesseract CLI and convert its TSV word records."""

    def __init__(
        self,
        executable: str | Path | None = None,
        *,
        language: str = "eng",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not language or any(character.isspace() for character in language):
            raise ValueError("language must be a non-empty Tesseract language code")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.executable = _find_tesseract(executable)
        self.language = language
        self.timeout_seconds = timeout_seconds
        try:
            result = subprocess.run(
                [self.executable, "--list-langs"],
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OCRBackendError(f"could not inspect Tesseract languages: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise OCRBackendError(f"Tesseract language query failed: {detail}")
        languages = {
            line.strip()
            for line in result.stdout.decode(errors="replace").splitlines()
            if line.strip() and not line.startswith("List of available languages")
        }
        if language not in languages:
            raise OCRBackendError(
                f"Tesseract language data {language!r} is unavailable"
            )

    def extract(self, image: np.ndarray) -> Iterable[OCRDetection]:
        try:
            encoded_ok, encoded_image = cv2.imencode(".png", image)
        except cv2.error as error:
            raise OCRBackendError(f"could not encode image for Tesseract: {error}") from error
        if not encoded_ok:
            raise OCRBackendError("could not encode image for Tesseract")

        try:
            result = subprocess.run(
                [
                    self.executable,
                    "stdin",
                    "stdout",
                    "-l",
                    self.language,
                    "--oem",
                    "1",
                    "--psm",
                    "6",
                    "tsv",
                ],
                input=encoded_image.tobytes(),
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OCRBackendError(f"Tesseract OCR could not process the image: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise OCRBackendError(f"Tesseract OCR failed: {detail}")

        try:
            rows = csv.DictReader(
                io.StringIO(result.stdout.decode("utf-8-sig", errors="strict")),
                delimiter="\t",
            )
            required_fields = {"level", "left", "top", "width", "height", "conf", "text"}
            if rows.fieldnames is None or not required_fields.issubset(rows.fieldnames):
                raise ValueError("Tesseract TSV is missing required fields")
            detections: list[OCRDetection] = []
            for row in rows:
                text = (row.get("text") or "").strip()
                if row.get("level") != "5" or not text:
                    continue
                left = float(row["left"])
                top = float(row["top"])
                width = float(row["width"])
                height = float(row["height"])
                raw_confidence = float(row["conf"])
                detections.append(
                    OCRDetection(
                        text=text,
                        confidence=(
                            raw_confidence / 100.0 if raw_confidence >= 0 else None
                        ),
                        bbox=(left, top, left + width, top + height),
                    )
                )
        except (csv.Error, KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
            raise OCRBackendError(f"could not parse Tesseract TSV output: {error}") from error
        return tuple(detections)


@dataclass(frozen=True)
class VideoTextObservation:
    text: str
    confidence: float | None
    timestamp: float
    frame_index: int | None
    bbox: tuple[float, float, float, float] | None
    source_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "source_id": self.source_id,
        }


class VideoTextExtractor:
    """Apply OCR to frames, suppress nearby repeated text, and bound retained history."""

    def __init__(
        self,
        backend: OCRBackend,
        *,
        history_size: int = 256,
        dedup_frame_gap: int = 15,
        dedup_time_gap_seconds: float = 2.0,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        if dedup_frame_gap < 0:
            raise ValueError("dedup_frame_gap cannot be negative")
        if dedup_time_gap_seconds < 0:
            raise ValueError("dedup_time_gap_seconds cannot be negative")
        self.backend = backend
        self.history_size = history_size
        self.dedup_frame_gap = dedup_frame_gap
        self.dedup_time_gap_seconds = dedup_time_gap_seconds
        self._observations: deque[VideoTextObservation] = deque(maxlen=history_size)
        self._last_seen: OrderedDict[tuple[str | None, str], VideoTextObservation] = OrderedDict()
        self.duplicates_suppressed = 0

    @property
    def observations(self) -> tuple[VideoTextObservation, ...]:
        return tuple(self._observations)

    def process(self, frame: PerceptionFrame) -> tuple[VideoTextObservation, ...]:
        emitted: list[VideoTextObservation] = []
        detections = self.backend.extract(frame.image)
        for detection in detections:
            if not isinstance(detection, OCRDetection):
                raise TypeError("OCR backend must return OCRDetection values")
            if not _normalize_text(detection.text):
                continue
            observation = VideoTextObservation(
                text=detection.text,
                confidence=detection.confidence,
                timestamp=frame.timestamp,
                frame_index=frame.frame_index,
                bbox=detection.bbox,
                source_id=frame.source_id or None,
            )
            key = (observation.source_id, _normalize_text(observation.text))
            previous = self._last_seen.get(key)
            duplicate = previous is not None and _is_nearby(
                previous,
                observation,
                max_frame_gap=self.dedup_frame_gap,
                max_time_gap=self.dedup_time_gap_seconds,
            )
            self._last_seen[key] = observation
            self._last_seen.move_to_end(key)
            while len(self._last_seen) > self.history_size:
                self._last_seen.popitem(last=False)

            if duplicate:
                self.duplicates_suppressed += 1
                continue
            self._observations.append(observation)
            emitted.append(observation)
        return tuple(emitted)


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _find_tesseract(executable: str | Path | None) -> str:
    if executable is not None:
        resolved = Path(executable).expanduser()
        if resolved.is_file():
            return str(resolved)
        raise OCRBackendError(f"Tesseract executable was not found: {resolved}")

    discovered = shutil.which("tesseract")
    if discovered:
        return discovered

    for environment_variable in ("ProgramFiles", "ProgramFiles(x86)"):
        install_root = os.environ.get(environment_variable)
        if install_root:
            candidate = Path(install_root) / "Tesseract-OCR" / "tesseract.exe"
            if candidate.is_file():
                return str(candidate)
    raise OCRBackendError(
        "Tesseract was not found; install the local engine or provide its executable path"
    )


def _is_nearby(
    previous: VideoTextObservation,
    current: VideoTextObservation,
    *,
    max_frame_gap: int,
    max_time_gap: float,
) -> bool:
    if current.timestamp < previous.timestamp:
        return False
    elapsed = current.timestamp - previous.timestamp
    if not math.isfinite(elapsed) or elapsed > max_time_gap:
        return False
    if previous.frame_index is not None and current.frame_index is not None:
        frame_gap = current.frame_index - previous.frame_index
        return 0 <= frame_gap <= max_frame_gap
    return True
