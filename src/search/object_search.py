from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.core.scene_state import SceneState


class TargetState(str, Enum):
    SEARCHING = "SEARCHING"
    FOUND = "FOUND"
    LOCKED = "LOCKED"
    LOST = "LOST"


@dataclass(frozen=True)
class SearchDetection:
    label: str
    confidence: float
    bounding_box: tuple[float, float, float, float] | None
    position: str
    normalized_horizontal: float


@dataclass
class TargetLock:
    label: str
    bounding_box: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    position: str | None = None
    normalized_horizontal: float = 0.5
    state: TargetState = TargetState.SEARCHING
    missed_frames: int = 0


class ObjectSearch:
    """Closed-vocabulary search and target lock using SceneState objects."""

    LABEL_ALIASES = {"phone": "cell phone"}

    def __init__(self, lost_after_frames: int = 3) -> None:
        self.lost_after_frames = max(1, lost_after_frames)
        self.target: TargetLock | None = None

    @classmethod
    def parse_query(cls, text: str) -> str | None:
        match = re.fullmatch(r"(?:find|locate)\s+(?:a |an |the )?(.+?)\s*[?.!]*", text.strip().lower())
        if not match:
            return None
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        return cls.LABEL_ALIASES.get(label, label) or None

    @classmethod
    def _normalize_label(cls, label: str) -> str:
        normalized = re.sub(r"\s+", " ", label.strip().lower())
        return cls.LABEL_ALIASES.get(normalized, normalized)

    @classmethod
    def detections_from_scene(cls, scene: SceneState) -> list[SearchDetection]:
        detections: list[SearchDetection] = []
        for obj in scene.objects:
            raw_box = getattr(obj, "bounding_box", None)
            try:
                bounding_box = tuple(float(value) for value in raw_box) if raw_box is not None else None
                if bounding_box is not None and len(bounding_box) != 4:
                    bounding_box = None
            except (TypeError, ValueError):
                bounding_box = None
            normalized_x = float(obj.normalized_horizontal)
            position = obj.position_category
            if position not in {"left", "center", "right"}:
                position = "left" if normalized_x < 1 / 3 else "right" if normalized_x > 2 / 3 else "center"
            detections.append(
                SearchDetection(
                    label=cls._normalize_label(obj.label),
                    confidence=float(obj.confidence),
                    bounding_box=bounding_box,
                    position=position,
                    normalized_horizontal=normalized_x,
                )
            )
        return detections

    @staticmethod
    def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _center_distance_ratio(
        first: tuple[float, float, float, float], second: tuple[float, float, float, float]
    ) -> float:
        first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
        second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
        first_diagonal = ((first[2] - first[0]) ** 2 + (first[3] - first[1]) ** 2) ** 0.5
        second_diagonal = ((second[2] - second[0]) ** 2 + (second[3] - second[1]) ** 2) ** 0.5
        diagonal = max(1.0, first_diagonal, second_diagonal)
        return ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5 / diagonal

    @staticmethod
    def _apply_detection(target: TargetLock, detection: SearchDetection) -> None:
        target.label = detection.label
        target.bounding_box = detection.bounding_box
        target.confidence = detection.confidence
        target.position = detection.position
        target.normalized_horizontal = detection.normalized_horizontal

    @classmethod
    def _matches(cls, target: TargetLock, detection: SearchDetection) -> bool:
        if target.bounding_box is not None and detection.bounding_box is not None:
            overlap = cls._iou(target.bounding_box, detection.bounding_box)
            distance = cls._center_distance_ratio(target.bounding_box, detection.bounding_box)
            return overlap >= 0.05 or distance <= 0.65
        return abs(target.normalized_horizontal - detection.normalized_horizontal) <= 0.25

    @classmethod
    def _match_score(cls, target: TargetLock, detection: SearchDetection) -> tuple[float, float, float]:
        if target.bounding_box is not None and detection.bounding_box is not None:
            overlap = cls._iou(target.bounding_box, detection.bounding_box)
            distance = cls._center_distance_ratio(target.bounding_box, detection.bounding_box)
            return overlap, -distance, detection.confidence
        distance = abs(target.normalized_horizontal - detection.normalized_horizontal)
        return 0.0, -distance, detection.confidence

    def begin_search(self, requested_label: str, scene: SceneState) -> TargetLock:
        requested_label = self._normalize_label(requested_label)
        self.target = TargetLock(label=requested_label)
        detections = self.detections_from_scene(scene)
        candidates = [detection for detection in detections if detection.label == requested_label]
        if not candidates:
            return self.target
        best = max(candidates, key=lambda detection: detection.confidence)
        self._apply_detection(self.target, best)
        self.target.state = TargetState.FOUND
        return self.target

    def update(self, scene: SceneState) -> TargetLock | None:
        if self.target is None or self.target.state is TargetState.LOST:
            return self.target
        detections = self.detections_from_scene(scene)
        candidates = [detection for detection in detections if detection.label == self.target.label]
        if self.target.state is TargetState.SEARCHING:
            if candidates:
                self._apply_detection(self.target, max(candidates, key=lambda detection: detection.confidence))
                self.target.state = TargetState.FOUND
            return self.target
        matches = [detection for detection in candidates if self._matches(self.target, detection)]
        if matches:
            best = max(matches, key=lambda detection: self._match_score(self.target, detection))
            self._apply_detection(self.target, best)
            self.target.missed_frames = 0
            self.target.state = TargetState.LOCKED
            return self.target
        self.target.missed_frames += 1
        if self.target.missed_frames >= self.lost_after_frames:
            self.target.state = TargetState.LOST
        return self.target
