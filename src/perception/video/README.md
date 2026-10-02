# Video Input Foundation

`VideoFrameSource` decodes a local video file into the existing `PerceptionFrame` abstraction. Each yielded frame carries the decoded source-frame index, monotonic timestamp, ndarray image, and derived width/height. Video decoding and sampling do not invoke perception models.

Sampling is configurable with `every_n_frames` (default 1) or `target_fps`; target-FPS sampling takes precedence when supplied. The source skips invalid decoded frames, reports missing/unreadable files with `VideoInputError`, yields no frames for an empty video, and ends cleanly at EOF. Call `close()` or use it as a context manager to release the decoder.

```python
from src.perception.video.video_source import VideoFrameSource

with VideoFrameSource("recording.mp4", target_fps=2.0) as source:
    for frame in source:
        output = pipeline.process(frame)
```

The pipeline remains responsible for validating and processing each `PerceptionFrame`. This foundation adds recorded-video input only; it does not implement scene segmentation, event extraction, video memory, or summarization.

## Temporal sampling foundation

`TemporalSampler` adds deterministic keyframe decisions and temporal metadata on top of sampled `PerceptionFrame` objects. It selects the first sample, samples with a meaningful downsampled pixel change, and a representative sample at the configured maximum interval. Each `TemporalFrameSample` retains the original frame, sample index, elapsed time/frame count, selection reason, and measured change. Its `TemporalFrameBuffer` retains only a configurable number of recent samples.

Pass `sample.frame` to the existing `PerceptionPipeline`; the sampler does not run perception, alter the frame abstraction, or infer events. Pixel-change thresholds are a simple visual heuristic, not semantic scene understanding.

## Semantic timeline

`SemanticTimeline` consumes the structured `SceneSnapshot` produced by the existing perception pipeline and converts tracker transitions and track states into bounded chronological event records. Track IDs are preserved, so same-label objects remain distinct. This is structured event data, not natural-language video understanding, event interpretation, or summarization.

## Video facts

`VideoFacts.from_timeline(...)` derives per-track facts and simple queries from the timeline's retained events only. It preserves event timestamps, frame indexes, track IDs, labels, positions, and confidence; derives observed first/last-seen timestamps, presence duration, appearance/reacquisition counts, and current presence; and supports queries for present, appeared, disappeared, moved, event timeline, and per-track history. Optional fact-history limits apply to retained timeline events, so facts cannot recover events already evicted and do not imply unobserved activity. The output is structured data only; it does not generate natural-language answers.

Run deterministic temporal tests with:

```powershell
D:\friday\.venv\Scripts\python.exe scripts/test_video_temporal.py
```

Run the mock-only input test with:

```powershell
D:\friday\.venv\Scripts\python.exe scripts/test_video_input.py
```
