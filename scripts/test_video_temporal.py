from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.types import PerceptionFrame
from src.perception.video.temporal import KeyframeSelector, TemporalSampler


def make_frame(index: int, image: np.ndarray) -> PerceptionFrame:
    return PerceptionFrame(
        image=image,
        timestamp=index / 10.0,
        source_id="video:mock",
        frame_index=index,
    )


def test_keyframes_follow_changes_and_max_interval() -> None:
    sampler = TemporalSampler(selector=KeyframeSelector(max_interval_frames=3))
    unchanged = np.zeros((24, 32, 3), dtype=np.uint8)
    changed = unchanged.copy()
    changed[:, :16] = 255

    samples = [
        sampler.process(make_frame(0, unchanged)),
        sampler.process(make_frame(1, unchanged.copy())),
        sampler.process(make_frame(2, changed)),
        sampler.process(make_frame(3, changed.copy())),
        sampler.process(make_frame(4, changed.copy())),
        sampler.process(make_frame(5, changed.copy())),
    ]
    assert [sample.is_keyframe for sample in samples] == [True, False, True, False, False, True]
    assert [sample.selection_reason for sample in samples] == [
        "first_frame",
        "no_significant_change",
        "meaningful_change",
        "no_significant_change",
        "no_significant_change",
        "maximum_interval",
    ]
    change = samples[2].change
    assert change is not None and change.meaningful
    assert change.mean_difference == 0.5
    assert change.changed_fraction == 0.5
    print("KEYFRAME SELECTION AND DETERMINISTIC CHANGE METRICS: PASS")


def test_temporal_metadata_and_bounded_buffer() -> None:
    sampler = TemporalSampler(
        selector=KeyframeSelector(max_interval_frames=10),
        buffer_capacity=2,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    first = sampler.process(make_frame(4, image))
    second = sampler.process(make_frame(7, image.copy()))
    third = sampler.process(make_frame(9, image.copy()))

    assert (first.sample_index, first.elapsed_seconds, first.elapsed_frames) == (0, 0.0, 0)
    assert second.sample_index == 1 and np.isclose(second.elapsed_seconds, 0.3)
    assert second.elapsed_frames == 3
    assert third.sample_index == 2 and np.isclose(third.elapsed_seconds, 0.2)
    assert third.elapsed_frames == 2
    assert len(sampler.buffer) == 2
    assert [sample.sample_index for sample in sampler.buffer.snapshot()] == [1, 2]
    assert [sample.frame.frame_index for sample in sampler.buffer.snapshot()] == [7, 9]
    print("TEMPORAL METADATA AND FIXED-CAPACITY BUFFER: PASS")


def main() -> None:
    test_keyframes_follow_changes_and_max_interval()
    test_temporal_metadata_and_bounded_buffer()
    print("VIDEO TEMPORAL FOUNDATION: PASS")


if __name__ == "__main__":
    main()
