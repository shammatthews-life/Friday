from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.friday import FridayAssistant
from src.core.scene_memory import SceneMemory
from src.core.scene_state import SceneObject, SceneState
from src.llm.tool_interface import CapabilityKind, CapabilityRegistry, CapabilityRequest
from src.llm.visionaid_bridge import SCENE_AWARENESS, VisionAidBridge
from src.perception.pipeline import PerceptionPipeline
from src.perception.pipeline import create_pipeline
from src.perception.depth_service import LazyDepthAnythingV2
from src.perception.spatial.relations import horizontal_region, vertical_region
from src.perception.tracker.iou_tracker import IoUTracker
from src.perception.types import Detection, PerceptionFrame
from src.safety.safety_engine import SafetyEngine, SafetyInput, SafetyState
from src.search.object_search import ObjectSearch, TargetState


IMAGE = np.indices((100, 100)).sum(axis=0).astype(np.uint8)
IMAGE = (IMAGE[:, :, None].repeat(3, axis=2) * 2) + 20


def make_detection(
    label: str,
    box: tuple[float, float, float, float],
    confidence: float = 0.9,
    width: int = 100,
    height: int = 100,
) -> Detection:
    left, top, right, bottom = box
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    normalized_x = center_x / width
    normalized_y = center_y / height
    return Detection(
        label=label,
        confidence=confidence,
        bounding_box=box,
        center_x=center_x,
        center_y=center_y,
        normalized_horizontal=normalized_x,
        normalized_vertical=normalized_y,
        position_category=horizontal_region(normalized_x),
        vertical_position=vertical_region(normalized_y),
    )


def make_frame(index: int, image: np.ndarray = IMAGE, source: str = "synthetic") -> PerceptionFrame:
    return PerceptionFrame(image=image, timestamp=float(index + 1), source_id=source, frame_index=index)


class MockDetector:
    def __init__(self, schedule: dict[int, list[Detection]]) -> None:
        self.schedule = schedule
        self.calls = 0
        self.load_time_ms = 0.0

    def detect(self, frame: PerceptionFrame) -> list[Detection]:
        self.calls += 1
        index = frame.frame_index or 0
        return [
            make_detection(
                item.label,
                item.bounding_box,
                item.confidence,
                frame.image.shape[1],
                frame.image.shape[0],
            )
            for item in self.schedule.get(index, [])
        ]


class MockDepth:
    def __init__(self, category: str = "relatively near", should_fail: bool = False) -> None:
        self.category = category
        self.should_fail = should_fail
        self.calls = 0
        self.load_time_ms = 0.0

    def estimate(self, frame: PerceptionFrame, detections: list[Detection]) -> dict[int, str]:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("mock depth unavailable")
        return {item.track_id: self.category for item in detections if item.track_id is not None}


def make_pipeline(
    schedule: dict[int, list[Detection]],
    *,
    depth: MockDepth | None = None,
    depth_interval: int = 1,
    max_missing: int = 3,
    reidentify: int = 8,
) -> tuple[PerceptionPipeline, MockDetector]:
    detector = MockDetector(schedule)
    pipeline = PerceptionPipeline(
        detector,
        depth_provider=depth,
        tracker=IoUTracker(
            max_missing_frames=max_missing,
            reidentify_frames=reidentify,
            max_center_distance=0.25,
            movement_threshold=0.08,
        ),
        depth_interval_frames=depth_interval,
    )
    return pipeline, detector


def test_single_multiple_and_duplicate_labels() -> None:
    pipeline, _ = make_pipeline(
        {
            0: [make_detection("bottle", (10, 20, 30, 50))],
            1: [
                make_detection("person", (5, 10, 20, 45)),
                make_detection("person", (75, 12, 92, 48)),
                make_detection("chair", (40, 55, 65, 90)),
            ],
        },
        depth=None,
    )
    first = pipeline.process(make_frame(0)).snapshot
    assert first.valid and len(first.scene_state.objects) == 1
    assert first.scene_state.objects[0].track_id == 1
    second = pipeline.process(make_frame(1)).snapshot
    people = [obj for obj in second.tracks if obj.label == "person" and obj.state == "visible"]
    assert len(second.scene_state.objects) == 3
    assert len(people) == 2 and people[0].track_id != people[1].track_id
    print("A ONE OBJECT: PASS")
    print("B MULTIPLE OBJECTS: PASS")
    print("C DUPLICATE LABELS KEEP SEPARATE TRACKS: PASS")
    pipeline.close()


def test_motion_jitter_disappearance_reacquisition() -> None:
    pipeline, _ = make_pipeline(
        {
            0: [make_detection("person", (35, 30, 55, 70))],
            1: [make_detection("person", (38, 30, 58, 70))],
            2: [make_detection("person", (52, 30, 72, 70))],
            5: [make_detection("person", (52, 30, 72, 70))],
        },
        depth=None,
        max_missing=2,
        reidentify=8,
    )
    first = pipeline.process(make_frame(0)).snapshot
    original_id = first.tracks[0].track_id
    jitter = pipeline.process(make_frame(1)).snapshot
    assert not any(event.kind == "moved" for event in jitter.scene_changes)
    assert jitter.tracks[0].state == "remained"
    moved = pipeline.process(make_frame(2)).snapshot
    assert any(event.kind == "moved" for event in moved.scene_changes)
    assert moved.tracks[0].track_id == original_id
    print("D MOVING OBJECT: PASS")
    print("I TINY JITTER DOES NOT CREATE A MOVE: PASS")

    pipeline.process(make_frame(3)).snapshot
    disappeared = pipeline.process(make_frame(4)).snapshot
    assert any(event.kind == "disappeared" for event in disappeared.scene_changes)
    assert any(track.state == "disappeared" for track in disappeared.tracks)
    reacquired = pipeline.process(make_frame(5)).snapshot
    assert any(event.kind == "reacquired" for event in reacquired.scene_changes)
    assert reacquired.scene_state.objects[0].track_id == original_id
    print("E DISAPPEARING OBJECT: PASS")
    print("F OBJECT REACQUISITION: PASS")
    print("I SCENE CHANGE EVENTS: PASS")
    pipeline.close()


def test_depth_positions_and_failures() -> None:
    depth = MockDepth("relatively near")
    pipeline, _ = make_pipeline(
        {
            0: [
                make_detection("bottle", (5, 5, 20, 25)),
                make_detection("chair", (42, 40, 58, 60)),
                make_detection("person", (80, 75, 96, 95)),
            ],
            1: [make_detection("bottle", (5, 5, 20, 25))],
            2: [make_detection("bottle", (5, 5, 20, 25))],
        },
        depth=depth,
        depth_interval=2,
    )
    first = pipeline.process(make_frame(0)).snapshot
    bottle = next(obj for obj in first.scene_state.objects if obj.label == "bottle")
    assert bottle.relative_depth_category == "unknown"
    assert depth.calls == 0
    assert [obj.position_category for obj in first.scene_state.objects] == ["left", "center", "right"]
    assert [obj.vertical_position for obj in first.scene_state.objects] == ["upper", "middle", "lower"]
    assert len(first.spatial_relations) > 0
    skipped = pipeline.process(make_frame(1)).snapshot
    assert depth.calls == 1
    assert skipped.scene_state.objects[0].relative_depth_category == "relatively near"
    skipped_again = pipeline.process(make_frame(2)).snapshot
    assert depth.calls == 1
    assert skipped_again.scene_state.objects[0].relative_depth_category == "relatively near"
    print("G RELATIVE DEPTH: PASS")
    print("DEPTH IS SCHEDULED, NOT CALLED EVERY FRAME: PASS")
    print("H LEFT/CENTER/RIGHT AND UPPER/MIDDLE/LOWER: PASS")
    print("SPATIAL RELATIONSHIPS: PASS")
    pipeline.close()

    missing_depth_pipeline, _ = make_pipeline(
        {0: [make_detection("bottle", (10, 10, 30, 30))]},
        depth=MockDepth(category="unknown"),
    )
    missing = missing_depth_pipeline.process(make_frame(0)).snapshot
    assert missing.scene_state.objects[0].relative_depth_category == "unknown"
    print("L MISSING DEPTH RESULT DEGRADES GRACEFULLY: PASS")
    missing_depth_pipeline.close()

    failing_depth_pipeline, _ = make_pipeline(
        {0: [make_detection("bottle", (10, 10, 30, 30))]},
        depth=MockDepth(should_fail=True),
    )
    failed = failing_depth_pipeline.process(make_frame(0)).snapshot
    assert failed.valid and len(failed.scene_state.objects) == 1
    assert failed.depth_summary.get("error") == "RuntimeError: mock depth unavailable"
    print("DEPTH FAILURE PRESERVES DETECTIONS: PASS")
    failing_depth_pipeline.close()


def test_retained_depth_estimator_reuses_model() -> None:
    factory_calls = 0

    class MockEstimator:
        def __init__(self, model_dir) -> None:
            nonlocal factory_calls
            factory_calls += 1
            self.estimate_calls = 0
            self.load_time_ms = 12.0

        def estimate(self, image_path):
            self.estimate_calls += 1
            return np.ones((10, 10), dtype=np.float32)

        def relative_depth_for_box(self, depth_map, bounding_box):
            return SimpleNamespace(value=0.5, category="relatively near")

    provider = LazyDepthAnythingV2(
        ROOT / "models/depth/depth-anything-v2-small",
        retain_model=True,
        estimator_factory=MockEstimator,
    )
    detection = make_detection("bottle", (10, 10, 30, 30))
    detection.track_id = 5
    first = provider.estimate(make_frame(10), [detection])
    second = provider.estimate(make_frame(11), [detection])
    assert first == second == {5: "relatively near"}
    assert factory_calls == 1
    assert provider.estimator.estimate_calls == 2
    provider.close()
    assert provider.estimator is None
    print("RETAINED DEPTH ESTIMATOR REUSES MODEL AND CLOSE RELEASES IT: PASS")


def test_search_and_ambiguity() -> None:
    search = ObjectSearch(lost_after_frames=2)
    initial = SceneState(
        objects=[
            SceneObject("bottle", 0.92, normalized_horizontal=0.2, position_category="left", bounding_box=(10, 10, 30, 30), track_id=11),
            SceneObject("bottle", 0.84, normalized_horizontal=0.8, position_category="right", bounding_box=(70, 10, 90, 30), track_id=22),
        ],
        timestamp=1.0,
    )
    target = search.begin_search("bottle", initial)
    assert target.state is TargetState.FOUND and target.track_id == 11
    moved = SceneState(
        objects=[
            SceneObject("bottle", 0.70, normalized_horizontal=0.21, position_category="left", bounding_box=(11, 10, 31, 30), track_id=11),
            SceneObject("bottle", 0.99, normalized_horizontal=0.81, position_category="right", bounding_box=(71, 10, 91, 30), track_id=22),
        ],
        timestamp=2.0,
    )
    target = search.update(moved)
    assert target is not None and target.state is TargetState.LOCKED and target.track_id == 11
    reacquired_with_new_id = SceneState(
        objects=[
            SceneObject(
                "bottle",
                0.86,
                normalized_horizontal=0.22,
                position_category="left",
                bounding_box=(12, 10, 32, 30),
                track_id=99,
            )
        ],
        timestamp=2.5,
    )
    target = search.update(reacquired_with_new_id)
    assert target is not None and target.state is TargetState.LOCKED and target.track_id == 99
    empty = SceneState(timestamp=3.0)
    search.update(empty)
    lost = search.update(empty)
    assert lost is not None and lost.state is TargetState.LOST
    print("J SEARCH TRACK-ID LOCK, MOVEMENT, AND LOSS: PASS")

    ambiguous_scene = SceneState(
        objects=[
            SceneObject("person", 0.9, position_category="left", normalized_horizontal=0.2),
            SceneObject("person", 0.9, position_category="right", normalized_horizontal=0.8),
        ],
        timestamp=1.0,
    )
    response = FridayAssistant().respond("Where is the person?", ambiguous_scene)
    assert "not sure" in response.lower()
    print("M AMBIGUOUS TARGET: PASS")


def test_scene_memory_uses_stable_track_identity() -> None:
    memory = SceneMemory(position_tolerance=0.15)
    memory.update(
        SceneState(
            objects=[
                SceneObject(
                    "bottle",
                    0.9,
                    normalized_horizontal=0.1,
                    position_category="left",
                    timestamp=1.0,
                    track_id=41,
                )
            ],
            timestamp=1.0,
        )
    )
    original_memory_id = memory.find_objects("bottle")[0].track_id
    memory.update(
        SceneState(
            objects=[
                SceneObject(
                    "bottle",
                    0.88,
                    normalized_horizontal=0.9,
                    position_category="right",
                    timestamp=2.0,
                    track_id=41,
                )
            ],
            timestamp=2.0,
        )
    )
    remembered = memory.find_objects("bottle")
    assert len(remembered) == 1
    assert remembered[0].track_id == original_memory_id
    assert remembered[0].position == "right"
    assert memory.get_newly_noticed_objects() == []
    print("SCENE MEMORY STABLE TRACK ID PREVENTS DUPLICATE: PASS")


def test_invalid_frames_safety_and_bridge_snapshot() -> None:
    pipeline, detector = make_pipeline({0: [make_detection("person", (20, 20, 40, 60))]})
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    rejected = pipeline.process(make_frame(0, blank)).snapshot
    assert not rejected.valid and detector.calls == 0
    malformed = pipeline.process(make_frame(1, np.empty((0, 0, 3), dtype=np.uint8))).snapshot
    assert not malformed.valid and detector.calls == 0
    print("K INVALID/DARK/BLANK FRAME REJECTION: PASS")

    safety = SafetyEngine(initial_state=SafetyState.UNKNOWN)
    result = safety.evaluate(
        SafetyInput(
            obstacle_present=None,
            relative_depth="unknown",
            confidence=None,
            target_is_tracked=None,
            timestamp=1.0,
        )
    )
    assert result.state is SafetyState.UNKNOWN
    print("N SAFETY UNKNOWN REMAINS AUTHORITATIVE: PASS")
    pipeline.close()

    full_pipeline, _ = make_pipeline(
        {
            0: [
                make_detection("person", (5, 20, 25, 70)),
                make_detection("bottle", (70, 20, 90, 60)),
            ]
        },
        depth=MockDepth("relatively far"),
    )
    output = full_pipeline.process(make_frame(0, source="video:sample#0"))
    assert output.snapshot.valid
    assert output.snapshot.source_id == "video:sample#0"
    assert full_pipeline.scene_memory.get_visible_objects()
    registry = VisionAidBridge(lambda: output.snapshot).register(CapabilityRegistry())
    result = registry.invoke(
        CapabilityRequest(SCENE_AWARENESS, CapabilityKind.INFORMATION)
    )
    assert result.data["perception"]["objects"]
    assert result.data["perception"]["spatial_relations"]
    print("O COMPLETE FRAME-TO-SCENE PIPELINE AND BRIDGE: PASS")
    full_pipeline.close()


def test_profiles_are_lazy() -> None:
    for profile in ("quality", "balanced", "realtime"):
        pipeline = create_pipeline(profile, root=ROOT)
        assert pipeline.detector.model is None
        assert pipeline.depth_provider is None or pipeline.depth_provider.estimator is None
        assert pipeline.depth_interval_frames >= 1
        pipeline.close()
    print("PROFILES LOAD MODELS LAZILY: PASS")


def main() -> None:
    test_single_multiple_and_duplicate_labels()
    test_motion_jitter_disappearance_reacquisition()
    test_depth_positions_and_failures()
    test_retained_depth_estimator_reuses_model()
    test_search_and_ambiguity()
    test_scene_memory_uses_stable_track_identity()
    test_invalid_frames_safety_and_bridge_snapshot()
    test_profiles_are_lazy()
    assert "ultralytics" not in sys.modules
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    print("NO MODEL RUNTIME LOADED: PASS")
    print("PERCEPTION PIPELINE TESTS: PASS")


if __name__ == "__main__":
    main()
