# Real Qwen grounded video smoke test

- Result: **PASS**
- Model: `models\llm\qwen3-8b\Qwen3-8B-Q4_K_M.gguf` (Qwen3-8B GGUF / Q4_K_M)
- Runtime: `0.00.001.398 I srv  llama_server: initializing ...
version: 0.5.0-dev (build 11146, commit 7fe450e19)
built with Clang 20.1.8 for Windows x86_64`
- Context size: 4096
- Verified local Qwen model identity: True
- Video: `data\test_videos\friday_video_smoke.avi`
- Session result: `completed`
- Sampled/processed frames: 4/4
- Qwen startup: 11.92037900001742
- Peak sampled process-tree RSS: 6749.8 MiB

## OCR/text observations

- No recognized text observations.

## FRIDAY responses and timings

### What happened in this video?

- Response: The video shows a sports ball moving across the scene. It first appears, then moves, and is reacquired at the end.
- First token: 52.43720340006985
- Completion: 56.73285400005989
- Total FRIDAY response: 56.73358060000464
- Context status: `grounded`
- Checks:
  - nonempty_real_model_response: PASS
  - no_internal_identifiers: PASS
  - no_invented_physical_measurements: PASS
  - full_video_context_supplied: PASS
  - visual_claims_have_supporting_events: PASS
  - visual_answer_uses_context_label: PASS
  - visual_events_have_matching_object_evidence: PASS
  - evidence_references_preserved: PASS

### What about the sports ball?

- Response: The sports ball first appeared, then moved, and was reacquired at the end of the video.
- First token: 46.68324589997064
- Completion: 50.01911979995202
- Total FRIDAY response: 50.020011899992824
- Context status: `grounded`
- Checks:
  - nonempty_real_model_response: PASS
  - no_internal_identifiers: PASS
  - no_invented_physical_measurements: PASS
  - sports_ball_track_context_supplied: PASS
  - sports_ball_track_is_unambiguous: PASS
  - sports_ball_identity_not_guessed: PASS

### When did it appear?

- Response: The sports ball appeared at the start of the video, which is at timestamp 0.0.
- First token: 47.05708009994123
- Completion: 50.15281439991668
- Total FRIDAY response: 50.153424700023606
- Context status: `grounded`
- Checks:
  - nonempty_real_model_response: PASS
  - no_internal_identifiers: PASS
  - no_invented_physical_measurements: PASS
  - followup_resolved_to_existing_track: PASS
  - followup_events_belong_to_track: PASS
  - timestamp_answer_uses_supplied_event_timestamp: PASS

### What text was shown?

- Response: No text was recognized in the video. Whether text was present cannot be determined.
- First token: 8.592633799999021
- Completion: 10.444684199988842
- Total FRIDAY response: 10.44499910005834
- Context status: `insufficient_context`
- Checks:
  - nonempty_real_model_response: PASS
  - no_internal_identifiers: PASS
  - no_invented_physical_measurements: PASS
  - ocr_context_states_no_recognized_text: PASS
  - ocr_does_not_claim_video_has_no_text: PASS
  - ocr_acknowledges_recognition_limit: PASS
  - ocr_context_contains_grounded_observations: PASS
- Recognized text supplied in this turn's context:
  - None.

## Failures

- None

## Notes

- Uses the existing local Qwen3-8B Q4_K_M adapter and its verified loopback llama.cpp server.
- Uses the repository's short synthetic AVI, realtime YOLO26n profile, existing Tesseract backend, and existing video session/context path.
- RSS is sampled process-tree resident memory using the existing `psutil` dependency; sampling is approximate.
- Detector labels describe model outputs on synthetic geometry and are not ground-truth object labels.
- No real-world or long-video inference was run.
