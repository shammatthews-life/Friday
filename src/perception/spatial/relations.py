from __future__ import annotations

from src.perception.types import Detection, SpatialRelation, TrackedObject


_DEPTH_ORDER = {
    "relatively far": 0,
    "relatively middle-distance": 1,
    "relatively near": 2,
}


def horizontal_region(normalized_x: float) -> str:
    if normalized_x < 1 / 3:
        return "left"
    if normalized_x > 2 / 3:
        return "right"
    return "center"


def vertical_region(normalized_y: float) -> str:
    if normalized_y < 1 / 3:
        return "upper"
    if normalized_y > 2 / 3:
        return "lower"
    return "middle"


def compute_spatial_relations(
    tracks: tuple[TrackedObject, ...] | list[TrackedObject],
    frame_width: int,
    frame_height: int,
    *,
    minimum_gap_ratio: float = 0.02,
) -> tuple[SpatialRelation, ...]:
    """Return only clear box-separated or category-separated relations."""
    if frame_width <= 0 or frame_height <= 0:
        return ()
    visible = [
        track
        for track in tracks
        if track.state in {"visible", "remained", "moved", "reacquired"}
    ]
    relations: list[SpatialRelation] = []
    for first_index, first in enumerate(visible):
        left_a, top_a, right_a, bottom_a = first.bounding_box
        for second in visible[first_index + 1 :]:
            left_b, top_b, right_b, bottom_b = second.bounding_box
            horizontal_gap_ab = left_b - right_a
            horizontal_gap_ba = left_a - right_b
            if horizontal_gap_ab / frame_width >= minimum_gap_ratio:
                relations.append(
                    SpatialRelation(
                        first.track_id,
                        first.label,
                        "left_of",
                        second.track_id,
                        second.label,
                        min(1.0, 0.6 + horizontal_gap_ab / frame_width),
                        "non-overlapping bounding boxes",
                    )
                )
            elif horizontal_gap_ba / frame_width >= minimum_gap_ratio:
                relations.append(
                    SpatialRelation(
                        second.track_id,
                        second.label,
                        "left_of",
                        first.track_id,
                        first.label,
                        min(1.0, 0.6 + horizontal_gap_ba / frame_width),
                        "non-overlapping bounding boxes",
                    )
                )

            vertical_gap_ab = top_b - bottom_a
            vertical_gap_ba = top_a - bottom_b
            if vertical_gap_ab / frame_height >= minimum_gap_ratio:
                relations.append(
                    SpatialRelation(
                        first.track_id,
                        first.label,
                        "above",
                        second.track_id,
                        second.label,
                        min(1.0, 0.6 + vertical_gap_ab / frame_height),
                        "non-overlapping bounding boxes",
                    )
                )
            elif vertical_gap_ba / frame_height >= minimum_gap_ratio:
                relations.append(
                    SpatialRelation(
                        second.track_id,
                        second.label,
                        "above",
                        first.track_id,
                        first.label,
                        min(1.0, 0.6 + vertical_gap_ba / frame_height),
                        "non-overlapping bounding boxes",
                    )
                )

            depth_a = _DEPTH_ORDER.get(first.relative_depth_category)
            depth_b = _DEPTH_ORDER.get(second.relative_depth_category)
            if depth_a is not None and depth_b is not None and depth_a != depth_b:
                closer, farther = (first, second) if depth_a > depth_b else (second, first)
                relations.append(
                    SpatialRelation(
                        closer.track_id,
                        closer.label,
                        "closer_than",
                        farther.track_id,
                        farther.label,
                        0.65,
                        "relative depth categories differ",
                    )
                )
    return tuple(relations)


def annotate_detection_position(detection: Detection) -> None:
    detection.position_category = horizontal_region(detection.normalized_horizontal)
    detection.vertical_position = vertical_region(detection.normalized_vertical)
