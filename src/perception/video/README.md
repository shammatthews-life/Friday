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

## Video evidence index

`VideoEvidenceIndex.from_facts(...)` indexes event evidence from retained `VideoFacts`, including source/current frame and timestamp, available previous event frame/timestamp for the same track, positions, and confidence. Lookups are available by event, track ID, or the event's source frame index. Previous-event references are included only when an earlier event for that track is present; unavailable frame indexes and metadata remain absent. Index history is bounded and does not recover evicted timeline events.

## Event episodes

`VideoEventSummarizer` groups retained `VideoFacts` events by track into structured presence, activity, disappearance, reappearance, or interrupted-reappearance episodes. Episodes retain their source events, available positions/confidences, and matching evidence-index records; partial inputs are marked incomplete and episode history is bounded. Grouping reflects only explicit timeline events and does not add natural-language interpretation.

## Video query engine

`VideoQueryEngine` provides deterministic structured queries over a `SemanticTimeline`, its retained `VideoFacts`, matching `VideoEvidenceIndex`, and episode summaries. It supports object presence/appearance/disappearance/movement, track history and observed duration, event/episode time ranges, evidence lookup, and the retained timeline. Optional label filters are case-insensitive; duplicate labels remain separate by track ID. Missing metadata and empty results are returned as-is, with no generated natural-language answers.

## End-to-end analysis session

`VideoAnalysisSession` composes a frame source, `TemporalSampler`, existing `PerceptionPipeline`, timeline, facts, evidence index, episode summarizer, and query engine. `run()` returns a structured `VideoAnalysisResult` with input metadata, sampled/processed/invalid frame counts, component results, query access, status, and explicit processing errors. Invalid perception snapshots are skipped from the semantic timeline while processing continues; source-open failures, decoder errors, empty input, and partial results are surfaced in the result. Session output is structured data only.

Run deterministic temporal tests with:

```powershell
D:\friday\.venv\Scripts\python.exe scripts/test_video_temporal.py
```

Run the mock-only input test with:

```powershell
D:\friday\.venv\Scripts\python.exe scripts/test_video_input.py
```
