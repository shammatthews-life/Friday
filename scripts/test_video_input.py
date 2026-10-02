from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.types import PerceptionFrame
from src.perception.video.video_source import VideoFrameSource, VideoInputError


class MockVideoCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 10.0, opened: bool = True) -> None:
        self.frames = frames
        self.fps = fps
        self.opened = opened
        self.index = -1
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.index += 1
        if self.index >= len(self.frames):
            return False, None
        return True, self.frames[self.index]

    def get(self, property_id: int) -> float:
        if property_id == 5:
            return self.fps
        if property_id == 0:
            return max(0, self.index) / self.fps * 1000
        return 0.0

    def release(self) -> None:
        self.released = True


def make_image(index: int) -> np.ndarray:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 0] = index + 30
    image[::2, ::2, 1] = 100
    return image


def test_every_frame_and_metadata(path: Path) -> None:
    capture = MockVideoCapture([make_image(index) for index in range(6)])
    source = VideoFrameSource(path, capture_factory=lambda _: capture)
    frames = list(source)
    assert all(isinstance(frame, PerceptionFrame) for frame in frames)
    assert [frame.frame_index for frame in frames] == list(range(6))
    assert [frame.width for frame in frames] == [32] * 6
    assert [frame.height for frame in frames] == [24] * 6
    timestamps = [frame.timestamp for frame in frames]
    assert all(current > previous for previous, current in zip(timestamps, timestamps[1:]))
    assert abs(timestamps[-1] - 0.5) < 1e-9
    assert source.yielded_frames == 6
    assert capture.released
    try:
        next(source)
    except StopIteration:
        pass
    else:
        raise AssertionError("Expected clean end-of-video")
    print("EVERY-FRAME METADATA, MONOTONIC TIMESTAMPS, EOF: PASS")


def test_every_n_frames(path: Path) -> None:
    capture = MockVideoCapture([make_image(index) for index in range(7)])
    frames = list(VideoFrameSource(path, every_n_frames=3, capture_factory=lambda _: capture))
    assert [frame.frame_index for frame in frames] == [0, 3, 6]
    assert [frame.timestamp for frame in frames] == [0.0, 0.3, 0.6]
    print("EVERY-N-FRAMES SAMPLING: PASS")


def test_target_fps(path: Path) -> None:
    capture = MockVideoCapture([make_image(index) for index in range(11)], fps=10.0)
    frames = list(VideoFrameSource(path, target_fps=2.0, capture_factory=lambda _: capture))
    assert [frame.frame_index for frame in frames] == [0, 5, 10]
    assert [frame.timestamp for frame in frames] == [0.0, 0.5, 1.0]
    print("TARGET-FPS SAMPLING: PASS")


def test_invalid_and_empty_inputs(path: Path, directory: Path) -> None:
    capture = MockVideoCapture([make_image(0), np.empty((0, 0, 3), dtype=np.uint8), make_image(2)])
    source = VideoFrameSource(path, capture_factory=lambda _: capture)
    frames = list(source)
    assert [frame.frame_index for frame in frames] == [0, 2]
    assert source.skipped_invalid_frames == 1

    empty_capture = MockVideoCapture([])
    empty_source = VideoFrameSource(path, capture_factory=lambda _: empty_capture)
    assert list(empty_source) == []
    assert empty_capture.released

    missing_path = directory / "missing.avi"
    try:
        list(VideoFrameSource(missing_path))
    except VideoInputError:
        pass
    else:
        raise AssertionError("Expected missing/unreadable path to be reported")

    unopened_capture = MockVideoCapture([], opened=False)
    try:
        list(VideoFrameSource(path, capture_factory=lambda _: unopened_capture))
    except VideoInputError:
        pass
    else:
        raise AssertionError("Expected an unreadable video source to be reported")
    assert unopened_capture.released
    print("INVALID FRAME, EMPTY VIDEO, UNREADABLE VIDEO: PASS")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="friday_video_input_test_") as temporary_dir:
        directory = Path(temporary_dir)
        path = directory / "mock_video.avi"
        path.write_bytes(b"mock video placeholder")
        test_every_frame_and_metadata(path)
        test_every_n_frames(path)
        test_target_fps(path)
        test_invalid_and_empty_inputs(path, directory)
    print("VIDEO INPUT FOUNDATION: PASS")


if __name__ == "__main__":
    main()
