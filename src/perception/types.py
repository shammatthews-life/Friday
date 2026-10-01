from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.scene_state import SceneState


@dataclass(frozen=True)
class PerceptionFrame:
    """Image/video/camera frame with source identity and capture timestamp."""

    image: np.ndarray
    timestamp: float = field(default_factory=time.time)
    source_id: str = "unknown"
    frame_index: int | None = None


@dataclass
class Detection:
    label: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    center_x: float
    center_y: float
    normalized_horizontal: float
    normalized_vertical: float
    position_category: str
    vertical_position: str
    track_id: int | None = None
    relative_depth_category: str = "unknown"


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    label: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    normalized_horizontal: float
    normalized_vertical: float
    position_category: str
    vertical_position: str
    last_seen_timestamp: float
    missed_frames: int
    state: str
    movement: str = "unknown"
    relative_depth_category: str = "unknown"


@dataclass(frozen=True)
class TrackEvent:
    kind: str
    track_id: int
    label: str
    timestamp: float
    previous_box: tuple[float, float, float, float] | None = None
    current_box: tuple[float, float, float, float] | None = None
    movement: str | None = None


@dataclass(frozen=True)
class SpatialRelation:
    subject_track_id: int
    subject_label: str
    relation: str
    object_track_id: int
    object_label: str
    confidence: float
    evidence: str


@dataclass(frozen=True)
class SceneSnapshot:
    """Fused structured scene while preserving the existing SceneState API."""

    scene_state: SceneState
    tracks: tuple[TrackedObject, ...] = ()
    spatial_relations: tuple[SpatialRelation, ...] = ()
    depth_summary: dict[str, Any] = field(default_factory=dict)
    newly_detected: tuple[TrackEvent, ...] = ()
    disappeared: tuple[TrackEvent, ...] = ()
    scene_changes: tuple[TrackEvent, ...] = ()
    timestamp: float = 0.0
    source_id: str = "unknown"
    valid: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        def serialize_event(event: TrackEvent) -> dict[str, Any]:
            return {
                "kind": event.kind,
                "track_id": event.track_id,
                "label": event.label,
                "timestamp": event.timestamp,
                "previous_box": list(event.previous_box) if event.previous_box else None,
                "current_box": list(event.current_box) if event.current_box else None,
                "movement": event.movement,
            }

        return {
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "valid": self.valid,
            "objects": [
                {
                    "track_id": obj.track_id,
                    "label": obj.label,
                    "confidence": obj.confidence,
                    "bounding_box": list(obj.bounding_box),
                    "position": obj.position_category,
                    "vertical_position": obj.vertical_position,
                    "relative_depth": obj.relative_depth_category,
                    "state": obj.state,
                    "movement": obj.movement,
                }
                for obj in self.tracks
            ],
            "spatial_relations": [
                {
                    "subject_track_id": item.subject_track_id,
                    "subject": item.subject_label,
                    "relation": item.relation,
                    "object_track_id": item.object_track_id,
                    "object": item.object_label,
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                }
                for item in self.spatial_relations
            ],
            "depth_summary": dict(self.depth_summary),
            "newly_detected": [serialize_event(event) for event in self.newly_detected],
            "disappeared": [serialize_event(event) for event in self.disappeared],
            "scene_changes": [serialize_event(event) for event in self.scene_changes],
            "error": self.error,
        }
