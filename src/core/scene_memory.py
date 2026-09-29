from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.scene_state import SceneObject, SceneState


@dataclass
class RememberedObject:
    track_id: int
    label: str
    confidence: float
    position: str
    normalized_horizontal: float
    timestamp: float
    last_seen_update: int
    missed_updates: int = 0
    stale: bool = False


@dataclass(frozen=True)
class Disappearance:
    label: str
    position: str
    timestamp: float
    track_id: int


class SceneMemory:
    """Short-term scene state with position-based matching for repeated labels."""

    def __init__(
        self,
        retention_seconds: float = 5.0,
        miss_limit: int = 2,
        position_tolerance: float = 0.25,
    ) -> None:
        self.retention_seconds = max(0.0, retention_seconds)
        self.miss_limit = max(1, miss_limit)
        self.position_tolerance = max(0.0, position_tolerance)
        self._objects: list[RememberedObject] = []
        self._recently_disappeared: list[Disappearance] = []
        self._newly_noticed: list[RememberedObject] = []
        self._previous_scene_ids: set[int] = set()
        self._update_number = 0
        self._next_track_id = 1
        self._current_timestamp = 0.0

    @staticmethod
    def _normalize_label(label: str) -> str:
        return re.sub(r"\s+", " ", label.strip().lower())

    @staticmethod
    def _position(obj: SceneObject) -> str:
        if obj.position_category in {"left", "center", "right"}:
            return obj.position_category
        x = obj.normalized_horizontal
        return "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"

    def _scene_timestamp(self, scene: SceneState) -> float:
        timestamps = [scene.timestamp]
        timestamps.extend(obj.timestamp for obj in scene.objects)
        timestamp = max(timestamps)
        if timestamp <= 0:
            return float(self._update_number)
        return timestamp

    def _expire(self) -> None:
        cutoff = self._current_timestamp - self.retention_seconds
        self._objects = [obj for obj in self._objects if obj.timestamp >= cutoff]
        self._recently_disappeared = [
            item for item in self._recently_disappeared if item.timestamp >= cutoff
        ]

    def _make_remembered(self, obj: SceneObject) -> RememberedObject:
        remembered = RememberedObject(
            track_id=self._next_track_id,
            label=self._normalize_label(obj.label),
            confidence=float(obj.confidence),
            position=self._position(obj),
            normalized_horizontal=float(obj.normalized_horizontal),
            timestamp=self._current_timestamp,
            last_seen_update=self._update_number,
        )
        self._next_track_id += 1
        return remembered

    def update(self, scene: SceneState) -> None:
        self._update_number += 1
        self._current_timestamp = self._scene_timestamp(scene)
        self._expire()
        self._newly_noticed = []

        available = set(range(len(self._objects)))
        matches: list[tuple[SceneObject, int]] = []
        for detected in scene.objects:
            label = self._normalize_label(detected.label)
            x = float(detected.normalized_horizontal)
            candidates = [
                (abs(self._objects[index].normalized_horizontal - x), index)
                for index in available
                if self._objects[index].label == label
                and abs(self._objects[index].normalized_horizontal - x) <= self.position_tolerance
            ]
            if candidates:
                _, index = min(candidates)
                available.remove(index)
                matches.append((detected, index))
            else:
                remembered = self._make_remembered(detected)
                self._objects.append(remembered)
                matches.append((detected, len(self._objects) - 1))

        current_ids: set[int] = set()
        matched_ids: set[int] = set()
        for detected, index in matches:
            remembered = self._objects[index]
            was_in_previous_scene = remembered.track_id in self._previous_scene_ids
            remembered.confidence = float(detected.confidence)
            remembered.position = self._position(detected)
            remembered.normalized_horizontal = float(detected.normalized_horizontal)
            remembered.timestamp = self._current_timestamp
            remembered.last_seen_update = self._update_number
            remembered.missed_updates = 0
            remembered.stale = False
            current_ids.add(remembered.track_id)
            matched_ids.add(remembered.track_id)
            self._recently_disappeared = [
                item for item in self._recently_disappeared if item.track_id != remembered.track_id
            ]
            if not was_in_previous_scene:
                self._newly_noticed.append(remembered)

        for index, remembered in enumerate(self._objects):
            if index in available and index not in {match_index for _, match_index in matches}:
                remembered.missed_updates += 1
                if remembered.missed_updates >= self.miss_limit and not remembered.stale:
                    remembered.stale = True
                    self._recently_disappeared.append(
                        Disappearance(
                            label=remembered.label,
                            position=remembered.position,
                            timestamp=self._current_timestamp,
                            track_id=remembered.track_id,
                        )
                    )

        self._previous_scene_ids = current_ids
        self._expire()

    def get_visible_objects(self) -> list[RememberedObject]:
        cutoff = self._current_timestamp - self.retention_seconds
        return [
            obj for obj in self._objects
            if not obj.stale and obj.missed_updates < self.miss_limit and obj.timestamp >= cutoff
        ]

    def find_objects(self, label: str) -> list[RememberedObject]:
        normalized = self._normalize_label(label)
        return [obj for obj in self.get_visible_objects() if obj.label == normalized]

    def count(self, label: str) -> int:
        return len(self.find_objects(label))

    def was_recently_seen(self, label: str) -> bool:
        normalized = self._normalize_label(label)
        cutoff = self._current_timestamp - self.retention_seconds
        return any(obj.label == normalized and obj.timestamp >= cutoff for obj in self._objects)

    def get_newly_noticed_objects(self) -> list[RememberedObject]:
        return list(self._newly_noticed)

    def get_recently_disappeared_objects(self) -> list[Disappearance]:
        cutoff = self._current_timestamp - self.retention_seconds
        return [item for item in self._recently_disappeared if item.timestamp >= cutoff]
