from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from src.core.scene_memory import SceneMemory
from src.core.scene_state import SceneState
from src.perception.depth_service import RelativeDepthProvider
from src.perception.detector.ultralytics_detector import Detector
from src.perception.frame import validate_frame
from src.perception.scene_fusion import SceneFusion
from src.perception.tracker.iou_tracker import IoUTracker
from src.perception.types import PerceptionFrame, SceneSnapshot


@dataclass(frozen=True)
class PipelineTiming:
    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    depth_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0
    detector_load_ms: float | None = None
    depth_load_ms: float | None = None


@dataclass(frozen=True)
class PipelineOutput:
    snapshot: SceneSnapshot
    timing: PipelineTiming


class PerceptionPipeline:
    """Sequential detection, tracking, scheduled relative depth, and fusion."""

    def __init__(
        self,
        detector: Detector,
        *,
        depth_provider: RelativeDepthProvider | None = None,
        tracker: IoUTracker | None = None,
        scene_memory: SceneMemory | None = None,
        depth_interval_frames: int = 5,
        profile_name: str = "balanced",
        minimum_brightness: float = 8.0,
        minimum_contrast: float = 1.0,
        minimum_spatial_gap_ratio: float = 0.02,
    ) -> None:
        self.detector = detector
        self.depth_provider = depth_provider
        self.tracker = tracker or IoUTracker()
        self.fusion = SceneFusion(
            scene_memory,
            minimum_spatial_gap_ratio=minimum_spatial_gap_ratio,
        )
        self.depth_interval_frames = max(1, depth_interval_frames)
        self.profile_name = profile_name
        self.minimum_brightness = minimum_brightness
        self.minimum_contrast = minimum_contrast
        self.frame_count = 0
        self._depth_by_track: dict[int, str] = {}

    @property
    def scene_memory(self) -> SceneMemory:
        return self.fusion.scene_memory

    def process(self, frame: PerceptionFrame) -> PipelineOutput:
        total_started = time.perf_counter()
        validation = validate_frame(
            frame,
            minimum_brightness=self.minimum_brightness,
            minimum_contrast=self.minimum_contrast,
        )
        if not validation.valid:
            snapshot = SceneSnapshot(
                scene_state=SceneState(timestamp=frame.timestamp),
                timestamp=frame.timestamp,
                source_id=frame.source_id,
                valid=False,
                error=validation.reason,
            )
            return PipelineOutput(
                snapshot,
                PipelineTiming(total_ms=(time.perf_counter() - total_started) * 1000),
            )

        self.frame_count += 1
        detection_started = time.perf_counter()
        try:
            detections = self.detector.detect(frame)
        except Exception as error:
            snapshot = SceneSnapshot(
                scene_state=SceneState(timestamp=frame.timestamp),
                timestamp=frame.timestamp,
                source_id=frame.source_id,
                valid=False,
                error=f"detector failed: {type(error).__name__}: {error}",
            )
            return PipelineOutput(
                snapshot,
                PipelineTiming(
                    detection_ms=(time.perf_counter() - detection_started) * 1000,
                    total_ms=(time.perf_counter() - total_started) * 1000,
                    detector_load_ms=getattr(self.detector, "load_time_ms", None),
                ),
            )
        detection_ms = (time.perf_counter() - detection_started) * 1000

        tracking_started = time.perf_counter()
        tracking = self.tracker.update(detections, frame.timestamp)
        tracking_ms = (time.perf_counter() - tracking_started) * 1000

        depth_ms = 0.0
        depth_error = None
        depth_load_ms = getattr(self.depth_provider, "load_time_ms", None)
        should_estimate_depth = (
            self.depth_provider is not None
            and bool(detections)
            and self.frame_count % self.depth_interval_frames == 0
        )
        if should_estimate_depth:
            depth_started = time.perf_counter()
            try:
                current_depth = self.depth_provider.estimate(frame, detections)
                self._depth_by_track.update(current_depth)
                self._depth_by_track = {
                    track.track_id: self._depth_by_track[track.track_id]
                    for track in tracking.tracks
                    if track.track_id in self._depth_by_track
                }
            except Exception as error:
                self._depth_by_track.clear()
                depth_error = f"{type(error).__name__}: {error}"
            depth_ms = (time.perf_counter() - depth_started) * 1000
            depth_load_ms = getattr(self.depth_provider, "load_time_ms", depth_load_ms)

        fusion_started = time.perf_counter()
        snapshot = self.fusion.compose(
            frame,
            detections,
            tracking,
            self._depth_by_track,
            depth_error=depth_error,
        )
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        timing = PipelineTiming(
            detection_ms=detection_ms,
            tracking_ms=tracking_ms,
            depth_ms=depth_ms,
            fusion_ms=fusion_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
            detector_load_ms=getattr(self.detector, "load_time_ms", None),
            depth_load_ms=depth_load_ms,
        )
        return PipelineOutput(snapshot, timing)

    def close(self) -> None:
        close_detector = getattr(self.detector, "close", None)
        if callable(close_detector):
            close_detector()
        close_depth = getattr(self.depth_provider, "close", None)
        if callable(close_depth):
            close_depth()

    def __enter__(self) -> PerceptionPipeline:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def create_pipeline(
    profile_name: str | None = None,
    *,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> PerceptionPipeline:
    """Build a lightweight pipeline; model objects remain unloaded until use."""
    import yaml

    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    settings_path = (
        Path(config_path)
        if config_path is not None
        else project_root / "configs/perception.yaml"
    )
    with settings_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    selected_profile = profile_name or config.get("default_profile", "balanced")
    profile = config.get("profiles", {}).get(selected_profile)
    if profile is None:
        raise ValueError(f"Unknown perception profile: {selected_profile}")

    from src.perception.detector.ultralytics_detector import UltralyticsDetector
    from src.perception.tracker.iou_tracker import IoUTracker

    detector_path = (project_root / profile["model_path"]).resolve()
    detector = UltralyticsDetector(
        detector_path,
        image_size=profile["image_size"],
        confidence=profile["confidence"],
        device=profile["device"],
    )
    depth_provider = None
    if profile.get("depth_enabled", True):
        from src.perception.depth_service import LazyDepthAnythingV2

        depth_path = (project_root / config["depth"]["model_dir"]).resolve()
        depth_provider = LazyDepthAnythingV2(
            depth_path,
            retain_model=bool(profile.get("retain_depth_model", False)),
        )

    tracking = config.get("tracking", {})
    tracker = IoUTracker(
        max_missing_frames=tracking.get("max_missing_frames", 3),
        reidentify_frames=tracking.get("reidentify_frames", 15),
        min_iou=tracking.get("minimum_iou", 0.03),
        max_center_distance=tracking.get("maximum_center_distance", 0.20),
        movement_threshold=tracking.get("movement_threshold", 0.08),
    )
    memory = SceneMemory(
        retention_seconds=config.get("scene_memory", {}).get("retention_seconds", 5.0),
        miss_limit=config.get("scene_memory", {}).get("miss_limit", 2),
        position_tolerance=config.get("scene_memory", {}).get("position_tolerance", 0.25),
    )
    return PerceptionPipeline(
        detector,
        depth_provider=depth_provider,
        tracker=tracker,
        scene_memory=memory,
        depth_interval_frames=profile.get("depth_interval_frames", 5),
        profile_name=selected_profile,
        minimum_brightness=config.get("depth", {}).get("minimum_brightness", 8.0),
        minimum_contrast=config.get("depth", {}).get("minimum_contrast", 1.0),
        minimum_spatial_gap_ratio=config.get("spatial", {}).get("minimum_gap_ratio", 0.02),
    )
