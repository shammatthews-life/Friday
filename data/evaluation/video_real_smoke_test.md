# Local video end-to-end smoke validation

- Result: `completed`
- Source: local deterministic synthetic clip (`data\test_videos\friday_video_smoke.avi`)
- Codec: MJPG / AVI
- Dimensions: 320 x 240
- FPS: 4.0
- Decoded frames: 8
- Approximate duration (decoded frames / FPS): 2.0 seconds
- Sampled frames: 4
- Processed frames: 4
- Invalid perception frames: 0
- Skipped invalid source frames: 0
- Temporal timestamps monotonic: True
- Sample frame indexes: `[0, 2, 4, 6]`
- Valid perception outputs: 4
- Valid outputs with no detections: 1
- Detected labels in valid snapshots: `['sports ball']`

## Temporal samples

`[{"frame_index": 0, "timestamp": 0.0, "is_keyframe": true, "selection_reason": "first_frame"}, {"frame_index": 2, "timestamp": 0.5, "is_keyframe": false, "selection_reason": "no_significant_change"}, {"frame_index": 4, "timestamp": 1.0, "is_keyframe": false, "selection_reason": "no_significant_change"}, {"frame_index": 6, "timestamp": 1.5, "is_keyframe": false, "selection_reason": "no_significant_change"}]`

## Structured outputs

Timeline events: `[{"event_type": "OBJECT_APPEARED", "track_id": 1, "label": "sports ball", "frame_index": 0, "timestamp": 0.0}, {"event_type": "OBJECT_MOVED", "track_id": 1, "label": "sports ball", "frame_index": 2, "timestamp": 0.5}, {"event_type": "OBJECT_REACQUIRED", "track_id": 1, "label": "sports ball", "frame_index": 6, "timestamp": 1.5}]`

Facts: `[{"track_id": 1, "label": "sports ball", "first_seen_timestamp": 0.0, "last_seen_timestamp": 1.5, "presence_duration_seconds": 1.5, "appearance_count": 1, "reacquisition_count": 1, "is_present": true}]`

Episodes: `[{"episode_type": "interrupted_reappearance", "track_id": 1, "label": "sports ball", "start_timestamp": 0.0, "end_timestamp": null, "event_types": ["OBJECT_APPEARED", "OBJECT_MOVED", "OBJECT_REACQUIRED"], "evidence_count": 3, "incomplete": true}]`

Queries: `{"objects_present": [{"track_id": 1, "label": "sports ball"}], "objects_appeared": [{"track_id": 1, "label": "sports ball"}], "timeline_event_count": 3}`

Evidence records: 3; source frame/timestamp references consistent: True.

## Limitations and issues

- No repository video was available, so this used a locally generated synthetic clip; it is not real-world video validation.
- YOLO26n (local realtime/CPU profile) labeled a geometric shape as `sports ball`. This is likely a false positive and must not be treated as ground truth or accuracy evidence.
- Only 4 of 8 frames were sampled. Depth inference is scheduled every 15 pipeline frames and was not invoked in this 4-frame run.
- Decoder/session errors: `[]`; source decoder error: `None`.
- No natural-language interpretation was generated.
