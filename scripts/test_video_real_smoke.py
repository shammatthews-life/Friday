from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.pipeline import PipelineOutput, create_pipeline
from src.perception.video.session import VideoAnalysisSession
from src.perception.video.temporal import TemporalSampler
from src.perception.video.video_source import VideoFrameSource


VIDEO_PATH = ROOT / "data" / "test_videos" / "friday_video_smoke.avi"
REPORT_PATH = ROOT / "data" / "evaluation" / "video_real_smoke_test.md"
VIDEO_FPS = 4.0
VIDEO_FRAME_COUNT = 8
FRAME_WIDTH = 320
FRAME_HEIGHT = 240


class ObservedPipeline:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.outputs = []

    def process(self, frame):
        output = self.pipeline.process(frame)
        self.outputs.append(output)
        return output

    def close(self) -> None:
        self.pipeline.close()


def create_synthetic_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        VIDEO_FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create local synthetic video: {path}")
    try:
        for frame_index in range(VIDEO_FRAME_COUNT):
            image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            image[:, :, 0] = np.arange(FRAME_WIDTH, dtype=np.uint8)[None, :] // 3 + 20
            image[:, :, 1] = np.arange(FRAME_HEIGHT, dtype=np.uint8)[:, None] // 3 + 25

            # A moving geometric shape makes decoded-frame changes deterministic.
            left = 20 + frame_index * 14
            cv2.rectangle(image, (left, 70), (left + 42, 190), (40, 70, 230), -1)
            cv2.circle(image, (left + 21, 52), 20, (220, 180, 30), -1)
            cv2.line(image, (0, 220), (FRAME_WIDTH - 1, 220), (230, 230, 230), 3)
            writer.write(image)
    finally:
        writer.release()


def _write_report(
    result,
    source: VideoFrameSource,
    sampler: TemporalSampler,
    labels: list[str],
    outputs: list[PipelineOutput],
) -> None:
    samples = sampler.buffer.snapshot()
    sample_records = [
        {
            "frame_index": sample.frame.frame_index,
            "timestamp": sample.frame.timestamp,
            "is_keyframe": sample.is_keyframe,
            "selection_reason": sample.selection_reason,
        }
        for sample in samples
    ]
    event_records = [
        {
            "event_type": event.event_type.value,
            "track_id": event.track_id,
            "label": event.label,
            "frame_index": event.frame_index,
            "timestamp": event.timestamp,
        }
        for event in result.timeline.events
    ]
    fact_records = [
        {
            "track_id": item.track_id,
            "label": item.label,
            "first_seen_timestamp": item.first_seen_timestamp,
            "last_seen_timestamp": item.last_seen_timestamp,
            "presence_duration_seconds": item.presence_duration_seconds,
            "appearance_count": item.appearance_count,
            "reacquisition_count": item.reacquisition_count,
            "is_present": item.is_present,
        }
        for item in result.facts.objects
    ]
    episode_records = [
        {
            "episode_type": episode.episode_type,
            "track_id": episode.track_id,
            "label": episode.label,
            "start_timestamp": episode.start_timestamp,
            "end_timestamp": episode.end_timestamp,
            "event_types": [event_type.value for event_type in episode.event_types],
            "evidence_count": len(episode.evidence),
            "incomplete": episode.incomplete,
        }
        for episode in result.episodes
    ]
    query_records = {
        "objects_present": [
            {"track_id": item.track_id, "label": item.label}
            for item in result.queries.objects_present()
        ],
        "objects_appeared": [
            {"track_id": item.track_id, "label": item.label}
            for item in result.queries.objects_appeared()
        ],
        "timeline_event_count": len(result.queries.timeline()),
    }
    fps = source.fps
    duration = source.decoded_frames / fps if fps else None
    report = f"""# Local video end-to-end smoke validation

- Result: `{result.status}`
- Source: local deterministic synthetic clip (`{VIDEO_PATH.relative_to(ROOT)}`)
- Codec: MJPG / AVI
- Dimensions: {FRAME_WIDTH} x {FRAME_HEIGHT}
- FPS: {fps!r}
- Decoded frames: {source.decoded_frames}
- Approximate duration (decoded frames / FPS): {duration!r} seconds
- Sampled frames: {result.sampled_frame_count}
- Processed frames: {result.processed_frame_count}
- Invalid perception frames: {result.invalid_perception_frame_count}
- Skipped invalid source frames: {source.skipped_invalid_frames}
- Temporal timestamps monotonic: {all(a.frame.timestamp < b.frame.timestamp for a, b in zip(samples, samples[1:]))}
- Sample frame indexes: `{[sample.frame.frame_index for sample in samples]}`
- Valid perception outputs: {sum(output.snapshot.valid for output in outputs)}
- Valid outputs with no detections: {sum(output.snapshot.valid and not output.snapshot.scene_state.objects for output in outputs)}
- Detected labels in valid snapshots: `{sorted(set(labels))}`

## Temporal samples

`{json.dumps(sample_records)}`

## Structured outputs

Timeline events: `{json.dumps(event_records)}`

Facts: `{json.dumps(fact_records)}`

Episodes: `{json.dumps(episode_records)}`

Queries: `{json.dumps(query_records)}`

Evidence records: {len(result.evidence.events)}; source frame/timestamp references consistent: {_evidence_consistent(result)}.

## Limitations and issues

- No repository video was available, so this used a locally generated synthetic clip; it is not real-world video validation.
- YOLO26n (local realtime/CPU profile) labeled a geometric shape as `sports ball`. This is likely a false positive and must not be treated as ground truth or accuracy evidence.
- Only {result.sampled_frame_count} of {source.decoded_frames} frames were sampled. Depth inference is scheduled every 15 pipeline frames and was not invoked in this 4-frame run.
- Decoder/session errors: `{list(result.errors)}`; source decoder error: `{source.last_error!r}`.
- No natural-language interpretation was generated.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def _evidence_consistent(result) -> bool:
    return all(
        len(result.evidence.by_event(event)) == 1
        and result.evidence.by_event(event)[0].source_frame_index == event.frame_index
        and result.evidence.by_event(event)[0].timestamp == event.timestamp
        for event in result.facts.events
    )


def main() -> None:
    create_synthetic_video(VIDEO_PATH)
    pipeline = ObservedPipeline(create_pipeline("realtime", root=ROOT))
    sampler = TemporalSampler(buffer_capacity=16)
    source = VideoFrameSource(VIDEO_PATH, every_n_frames=2)
    try:
        result = VideoAnalysisSession(
            source,
            pipeline,
            temporal_sampler=sampler,
            history_size=64,
        ).run()
    finally:
        pipeline.close()

    samples = sampler.buffer.snapshot()
    if result.status != "completed":
        raise AssertionError(f"Expected completed local smoke run, got {result.status}: {result.errors}")
    if source.last_error:
        raise AssertionError(f"Video decoder reported an error: {source.last_error}")
    if source.decoded_frames != VIDEO_FRAME_COUNT:
        raise AssertionError(f"Expected {VIDEO_FRAME_COUNT} decoded frames, got {source.decoded_frames}")
    if result.sampled_frame_count != 4 or result.processed_frame_count != 4:
        raise AssertionError(
            f"Expected 4 sampled/processed frames, got {result.sampled_frame_count}/{result.processed_frame_count}"
        )
    if len(samples) != result.sampled_frame_count:
        raise AssertionError("Temporal sampler sample count does not match session")
    if any(not output.snapshot.valid for output in pipeline.outputs):
        raise AssertionError("Perception pipeline returned an invalid snapshot")
    if not all(
        current.frame.timestamp > previous.frame.timestamp
        for previous, current in zip(samples, samples[1:])
    ):
        raise AssertionError("Sample timestamps are not strictly increasing")
    if not all(math.isfinite(sample.frame.timestamp) for sample in samples):
        raise AssertionError("A sampled frame timestamp is not finite")
    if not all(sample.frame.frame_index is not None for sample in samples):
        raise AssertionError("A sampled frame is missing its decoded source index")
    if [sample.frame.frame_index for sample in samples] != [0, 2, 4, 6]:
        raise AssertionError("Every-other-frame sampling returned unexpected source indexes")
    if not _evidence_consistent(result):
        raise AssertionError("Evidence references do not match retained timeline event metadata")
    if any(event not in result.timeline.events for event in result.facts.events):
        raise AssertionError("Facts contain events that are absent from the source timeline")

    labels = sorted(
        {
            item.label
            for output in pipeline.outputs
            if output.snapshot.valid
            for item in output.snapshot.scene_state.objects
        }
    )
    _write_report(result, source, sampler, labels, pipeline.outputs)
    print(f"VIDEO: {VIDEO_PATH.relative_to(ROOT)}")
    print(f"DECODED: {source.decoded_frames}; FPS: {source.fps}; duration: {source.decoded_frames / source.fps if source.fps else 'unknown'}")
    print(f"SAMPLED/PROCESSED: {result.sampled_frame_count}/{result.processed_frame_count}")
    print(f"DETECTED LABELS: {labels or 'none'}")
    print(f"TIMELINE EVENTS: {len(result.timeline.events)}")
    print(f"FACTS / EVIDENCE / EPISODES: {len(result.facts.objects)} / {len(result.evidence.events)} / {len(result.episodes)}")
    print(f"QUERY PRESENT OBJECTS: {[(item.track_id, item.label) for item in result.queries.objects_present()]}")
    print(f"REPORT: {REPORT_PATH.relative_to(ROOT)}")
    print("REAL VIDEO PIPELINE SMOKE: PASS")


if __name__ == "__main__":
    main()
