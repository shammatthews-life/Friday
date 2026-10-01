from __future__ import annotations

from dataclasses import dataclass
import math

from src.perception.types import Detection, TrackEvent, TrackedObject


@dataclass
class _Track:
    track_id: int
    label: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    normalized_horizontal: float
    normalized_vertical: float
    position_category: str
    vertical_position: str
    last_seen_timestamp: float
    missed_frames: int = 0
    state: str = "visible"
    movement: str = "unknown"
    relative_depth_category: str = "unknown"
    disappeared_emitted: bool = False


@dataclass(frozen=True)
class TrackingUpdate:
    tracks: tuple[TrackedObject, ...]
    events: tuple[TrackEvent, ...]
    newly_detected: tuple[TrackEvent, ...]
    disappeared: tuple[TrackEvent, ...]


class IoUTracker:
    """Deterministic one-to-one tracker for modest object motion and occlusion."""

    def __init__(
        self,
        *,
        max_missing_frames: int = 3,
        reidentify_frames: int = 15,
        min_iou: float = 0.03,
        max_center_distance: float = 0.20,
        movement_threshold: float = 0.08,
    ) -> None:
        self.max_missing_frames = max(1, max_missing_frames)
        self.reidentify_frames = max(self.max_missing_frames, reidentify_frames)
        self.min_iou = max(0.0, min_iou)
        self.max_center_distance = max(0.0, max_center_distance)
        self.movement_threshold = max(0.0, movement_threshold)
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._frame_index = 0

    @staticmethod
    def _iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _association_score(self, track: _Track, detection: Detection) -> float | None:
        if track.label.strip().lower() != detection.label.strip().lower():
            return None
        iou = self._iou(track.bounding_box, detection.bounding_box)
        center_distance = math.hypot(
            track.normalized_horizontal - detection.normalized_horizontal,
            track.normalized_vertical - detection.normalized_vertical,
        )
        if iou < self.min_iou and center_distance > self.max_center_distance:
            return None
        return 0.65 * iou + 0.35 * max(0.0, 1.0 - center_distance / max(self.max_center_distance, 1e-6))

    def update(self, detections: list[Detection], timestamp: float) -> TrackingUpdate:
        self._frame_index += 1
        candidates: list[tuple[float, int, int]] = []
        eligible = [
            track
            for track in self._tracks.values()
            if track.missed_frames <= self.reidentify_frames
        ]
        for detection_index, detection in enumerate(detections):
            for track in eligible:
                score = self._association_score(track, detection)
                if score is not None:
                    candidates.append((score, track.track_id, detection_index))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        match_by_detection: dict[int, _Track] = {}
        for _, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            match_by_detection[detection_index] = self._tracks[track_id]

        events: list[TrackEvent] = []
        newly_detected: list[TrackEvent] = []
        disappeared: list[TrackEvent] = []

        for detection_index, detection in enumerate(detections):
            track = match_by_detection.get(detection_index)
            if track is None:
                track = _Track(
                    track_id=self._next_track_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    bounding_box=detection.bounding_box,
                    normalized_horizontal=detection.normalized_horizontal,
                    normalized_vertical=detection.normalized_vertical,
                    position_category=detection.position_category,
                    vertical_position=detection.vertical_position,
                    last_seen_timestamp=timestamp,
                    state="visible",
                    movement="stationary",
                    relative_depth_category=detection.relative_depth_category,
                )
                self._tracks[track.track_id] = track
                self._next_track_id += 1
                matched_tracks.add(track.track_id)
                matched_detections.add(detection_index)
                detection.track_id = track.track_id
                event = TrackEvent("appeared", track.track_id, track.label, timestamp, current_box=track.bounding_box)
                events.append(event)
                newly_detected.append(event)
                continue

            previous_box = track.bounding_box
            previous_center = (track.normalized_horizontal, track.normalized_vertical)
            was_missing = track.missed_frames > 0
            track.movement = self._movement(
                previous_center,
                (detection.normalized_horizontal, detection.normalized_vertical),
            )
            track.confidence = detection.confidence
            track.bounding_box = detection.bounding_box
            track.normalized_horizontal = detection.normalized_horizontal
            track.normalized_vertical = detection.normalized_vertical
            track.position_category = detection.position_category
            track.vertical_position = detection.vertical_position
            track.last_seen_timestamp = timestamp
            track.missed_frames = 0
            track.state = (
                "reacquired"
                if was_missing
                else "moved"
                if track.movement != "stationary"
                else "remained"
            )
            if detection.relative_depth_category != "unknown":
                track.relative_depth_category = detection.relative_depth_category
            detection.track_id = track.track_id
            if was_missing:
                events.append(
                    TrackEvent("reacquired", track.track_id, track.label, timestamp, previous_box, track.bounding_box)
                )
            elif track.movement != "stationary":
                events.append(
                    TrackEvent("moved", track.track_id, track.label, timestamp, previous_box, track.bounding_box, track.movement)
                )

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.missed_frames += 1
            track.state = (
                "disappeared"
                if track.missed_frames >= self.max_missing_frames
                else "missing"
            )
            track.movement = "unknown"
            if track.missed_frames == self.max_missing_frames and not track.disappeared_emitted:
                track.state = "disappeared"
                track.disappeared_emitted = True
                event = TrackEvent(
                    "disappeared",
                    track.track_id,
                    track.label,
                    timestamp,
                    previous_box=track.bounding_box,
                )
                events.append(event)
                disappeared.append(event)
            if track.missed_frames > self.reidentify_frames:
                del self._tracks[track_id]

        track_snapshots = tuple(self._snapshot(track) for track in self._tracks.values())
        return TrackingUpdate(
            tracks=track_snapshots,
            events=tuple(events),
            newly_detected=tuple(newly_detected),
            disappeared=tuple(disappeared),
        )

    def _movement(
        self,
        previous: tuple[float, float],
        current: tuple[float, float],
    ) -> str:
        delta_x = current[0] - previous[0]
        delta_y = current[1] - previous[1]
        if math.hypot(delta_x, delta_y) < self.movement_threshold:
            return "stationary"
        horizontal = "right" if delta_x > 0 else "left"
        vertical = "down" if delta_y > 0 else "up"
        if abs(delta_x) >= abs(delta_y) * 1.5:
            return horizontal
        if abs(delta_y) >= abs(delta_x) * 1.5:
            return vertical
        return f"{vertical}-{horizontal}"

    @staticmethod
    def _snapshot(track: _Track) -> TrackedObject:
        return TrackedObject(
            track_id=track.track_id,
            label=track.label,
            confidence=track.confidence,
            bounding_box=track.bounding_box,
            normalized_horizontal=track.normalized_horizontal,
            normalized_vertical=track.normalized_vertical,
            position_category=track.position_category,
            vertical_position=track.vertical_position,
            last_seen_timestamp=track.last_seen_timestamp,
            missed_frames=track.missed_frames,
            state=track.state,
            movement=track.movement,
            relative_depth_category=track.relative_depth_category,
        )
