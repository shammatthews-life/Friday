# Real Qwen grounded video smoke test

- Result: **PASS**
- Model: `models\llm\qwen3-8b\Qwen3-8B-Q4_K_M.gguf` (Qwen3-8B GGUF / Q4_K_M)
- Runtime: `0.00.001.360 I srv  llama_server: initializing ...
version: 0.5.0-dev (build 11146, commit 7fe450e19)
built with Clang 20.1.8 for Windows x86_64`
- Context size: 4096
- Verified local Qwen model identity: True
- Video: `data\test_videos\friday_video_smoke.avi`
- Session result: `completed`
- Sampled/processed frames: 4/4
- Qwen startup: 11.780120500014164
- Peak sampled process-tree RSS: 6197.1 MiB

## OCR/text observations

- No recognized text observations.

## FRIDAY responses and timings

### What happened in this video?

- Response: The video shows a sports ball moving across the scene. It first appears, then moves, and is later reacquired as it continues to move.
- First token: 33.41772199992556
- Prompt/context construction: 0.00018560001626610756 / 0.00011609995272010565 seconds
- Request setup / connection setup: 0.00010890001431107521 / 0.030432499945163727 seconds
- Estimated prompt processing: 33.3872894999804 seconds
- Completion generation: 4.214069499983452 seconds
- Request round-trip: 37.63179149990901 seconds
- Total FRIDAY response: 37.63230749999639
- Prompt/completion tokens: 2193 / 31
- Request bytes: 7222
- Prompt message character counts: system=1839, system=4710, user=28
- Context section character counts: selection=22, video_metadata=347, text_observation_status=183, time_ranges=109, visual_objects=988, visual_events=775, episodes=2366, text_observations=2, text_history=30, evidence_references=1118, truncated=2
- Context status: `grounded`
- Checks:
  - nonempty_real_model_response: PASS
  - no_internal_identifiers: PASS
  - no_invented_physical_measurements: PASS
  - no_absolute_text_absence_claim: PASS
  - full_video_context_supplied: PASS
  - visual_claims_have_supporting_events: PASS
  - visual_answer_uses_context_label: PASS
  - visual_events_have_matching_object_evidence: PASS
  - evidence_references_preserved: PASS

### What about the sports ball?

- Response: The sports ball first appeared, then moved, and was later reacquired as it continued to move across the scene.
- First token: 27.754758200026117
- Prompt/context construction: 0.0003635999746620655 / 0.0001362999901175499 seconds
- Request setup / connection setup: 0.00022599997464567423 / 0.21369640005286783 seconds
- Estimated prompt processing: 27.54106179997325 seconds
- Completion generation: 3.339654400013387 seconds
- Request round-trip: 31.094412600039504 seconds
- Total FRIDAY response: 31.09523860004265
- Prompt/completion tokens: 2246 / 25
- Request bytes: 7483
- Prompt message character counts: system=1839, user=28, assistant=133, system=4736, user=27
- Context section character counts: selection=48, video_metadata=347, text_observation_status=183, time_ranges=109, visual_objects=988, visual_events=775, episodes=2366, text_observations=2, text_history=30, evidence_references=1118, truncated=2
- Context status: `grounded`
- Checks:
  - nonempty_real_model_response: PASS
  - no_internal_identifiers: PASS
  - no_invented_physical_measurements: PASS
  - no_absolute_text_absence_claim: PASS
  - sports_ball_track_context_supplied: PASS
  - sports_ball_track_is_unambiguous: PASS
  - sports_ball_identity_not_guessed: PASS

### When did it appear?

- Response: The sports ball appeared at the start of the video, which is at timestamp 0.0.
- First token: 28.09913929994218
- Prompt/context construction: 0.00022309995256364346 / 0.0001351999817416072 seconds
- Request setup / connection setup: 0.0001428000396117568 / 0.33152080001309514 seconds
- Estimated prompt processing: 27.767618499929085 seconds
- Completion generation: 2.680962600046769 seconds
- Request round-trip: 30.78010189998895 seconds
- Total FRIDAY response: 30.780722100054845
- Prompt/completion tokens: 2284 / 21
- Request bytes: 7666
- Prompt message character counts: system=1839, user=28, assistant=133, user=27, assistant=110, system=4721, user=19
- Context section character counts: selection=33, video_metadata=347, text_observation_status=183, time_ranges=109, visual_objects=988, visual_events=775, episodes=2366, text_observations=2, text_history=30, evidence_references=1118, truncated=2
- Context status: `grounded`
- Checks:
  - nonempty_real_model_response: PASS
  - no_internal_identifiers: PASS
  - no_invented_physical_measurements: PASS
  - no_absolute_text_absence_claim: PASS
  - followup_resolved_to_existing_track: PASS
  - followup_events_belong_to_track: PASS
  - timestamp_answer_uses_supplied_event_timestamp: PASS

### What text was shown?

- Response: No text was recognized in the video. Whether text was present cannot be determined.
- First token: 8.546021500020288
- Prompt/context construction: 9.789993055164814e-05 / 4.199997056275606e-05 seconds
- Request setup / connection setup: 0.0001175999641418457 / 0.2767836001003161 seconds
- Estimated prompt processing: 8.269237899919972 seconds
- Completion generation: 1.9835664000129327 seconds
- Request round-trip: 10.52958790003322 seconds
- Total FRIDAY response: 10.52990810002666
- Prompt/completion tokens: 1031 / 17
- Request bytes: 5029
- Prompt message character counts: system=1839, user=28, assistant=133, user=27, assistant=110, user=19, assistant=78, system=2119, user=20
- Context section character counts: selection=20, video_metadata=347, text_observation_status=183, time_ranges=67, visual_objects=2, visual_events=2, episodes=2, text_observations=2, text_history=30, evidence_references=2, truncated=2
- Context status: `insufficient_context`
- Checks:
  - nonempty_real_model_response: PASS
  - no_internal_identifiers: PASS
  - no_invented_physical_measurements: PASS
  - no_absolute_text_absence_claim: PASS
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
