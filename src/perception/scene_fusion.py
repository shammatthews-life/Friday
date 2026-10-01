from __future__ import annotations

from dataclasses import replace
from src.core.scene_memory import SceneMemory
from src.core.scene_state import SceneObject, SceneState
from src.perception.spatial.relations import compute_spatial_relations
from src.perception.tracker.iou_tracker import TrackingUpdate
from src.perception.types import Detection, PerceptionFrame, SceneSnapshot, TrackedObject


class SceneFusion:
    """Fuse detections, tracker output, relative depth, memory, and geometry."""

    def __init__(
        self,
        scene_memory: SceneMemory | None = None,
        *,
        minimum_spatial_gap_ratio: float = 0.02,
    ) -> None:
        self.scene_memory = scene_memory or SceneMemory()
        self.minimum_spatial_gap_ratio = max(0.0, minimum_spatial_gap_ratio)

    def compose(
        self,
        frame: PerceptionFrame,
        detections: list[Detection],
        tracking: TrackingUpdate,
        depth_categories: dict[int, str],
        *,
        depth_error: str | None = None,
    ) -> SceneSnapshot:
        height, width = frame.image.shape[:2]
        enriched_detections: list[Detection] = []
        for detection in detections:
            category = depth_categories.get(detection.track_id or -1, "unknown")
            detection.relative_depth_category = category
            enriched_detections.append(detection)

        tracks: tuple[TrackedObject, ...] = tuple(
            replace(
                track,
                relative_depth_category=depth_categories.get(
                    track.track_id, track.relative_depth_category
                ),
            )
            for track in tracking.tracks
        )
        scene = SceneState(timestamp=frame.timestamp)
        for detection in enriched_detections:
            scene.add_object(
                SceneObject(
                    label=detection.label,
                    confidence=detection.confidence,
                    center_x=detection.center_x,
                    center_y=detection.center_y,
                    normalized_horizontal=detection.normalized_horizontal,
                    position_category=detection.position_category,
                    timestamp=frame.timestamp,
                    distance=detection.relative_depth_category,
                    bounding_box=detection.bounding_box,
                    track_id=detection.track_id,
                    normalized_vertical=detection.normalized_vertical,
                    vertical_position=detection.vertical_position,
                    relative_depth_category=detection.relative_depth_category,
                )
            )
        self.scene_memory.update(scene)
        relations = compute_spatial_relations(
            tracks,
            width,
            height,
            minimum_gap_ratio=self.minimum_spatial_gap_ratio,
        )
        depth_summary = self._depth_summary(tracks)
        if depth_error:
            depth_summary["error"] = depth_error
        return SceneSnapshot(
            scene_state=scene,
            tracks=tracks,
            spatial_relations=relations,
            depth_summary=depth_summary,
            newly_detected=tracking.newly_detected,
            disappeared=tracking.disappeared,
            scene_changes=tracking.events,
            timestamp=frame.timestamp,
            source_id=frame.source_id,
            valid=True,
            error=None,
        )

    @staticmethod
    def _depth_summary(tracks: tuple[TrackedObject, ...]) -> dict[str, Any]:
        counts = {category: 0 for category in (
            "relatively near",
            "relatively middle-distance",
            "relatively far",
            "unknown",
        )}
        for track in tracks:
            if track.state not in {"visible", "remained", "moved", "reacquired"}:
                continue
            category = track.relative_depth_category
            counts[category if category in counts else "unknown"] += 1
        known_categories = [key for key, count in counts.items() if count and key != "unknown"]
        return {
            "relative_only": True,
            "counts": counts,
            "ordering_available": len(known_categories) > 1,
        }
