# Real Qwen grounded video smoke test

- Result: **PASS**
- Model: `models\llm\qwen3-8b\Qwen3-8B-Q4_K_M.gguf` (Qwen3-8B GGUF / Q4_K_M)
- Runtime: `0.00.001.391 I srv  llama_server: initializing ...
version: 0.5.0-dev (build 11146, commit 7fe450e19)
built with Clang 20.1.8 for Windows x86_64`
- Context size: 4096
- Verified local Qwen model identity: True
- Video: `data\test_videos\friday_video_smoke.avi`
- Session result: `completed`
- Sampled/processed frames: 4/4
- Qwen startup: 11.359270700020716
- Peak sampled process-tree RSS: 6172.3 MiB

## OCR/text observations

- No recognized text observations.

## FRIDAY responses and timings

### What happened in this video?

- Response: The video shows a sports ball moving across the scene. It first appears, then moves, and is later reacquired as it continues to move.
- First token: 33.55996879993472
- Prompt/context construction: 0.0001903000520542264 / 0.00012510002125054598 seconds
- Request setup / connection setup: 0.00010950001887977123 / 0.03104869998060167 seconds
- Estimated prompt processing: 33.52892009995412 seconds
- Completion generation: 4.930233600083739 seconds
- Request round-trip: 38.49020240001846 seconds
- Total FRIDAY response: 38.490737100015394
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
- First token: 28.106568800052628
- Prompt/context construction: 0.00035660003777593374 / 0.00013649999164044857 seconds
- Request setup / connection setup: 0.00013529998250305653 / 0.21789329999592155 seconds
- Estimated prompt processing: 27.888675500056706 seconds
- Completion generation: 3.628538199933246 seconds
- Request round-trip: 31.735106999985874 seconds
- Total FRIDAY response: 31.735831100028008
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
- First token: 28.76504849991761
- Prompt/context construction: 0.00019150006119161844 / 0.00010010006371885538 seconds
- Request setup / connection setup: 0.00010710000060498714 / 0.2512423999141902 seconds
- Estimated prompt processing: 28.51380610000342 seconds
- Completion generation: 3.04871920007281 seconds
- Request round-trip: 31.81376769999042 seconds
- Total FRIDAY response: 31.81425619998481
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
- First token: 9.064427200006321
- Prompt/context construction: 0.00013429997488856316 / 6.330001633614302e-05 seconds
- Request setup / connection setup: 0.00011939997784793377 / 0.26147519994992763 seconds
- Estimated prompt processing: 8.802952000056393 seconds
- Completion generation: 2.1914264999795705 seconds
- Request round-trip: 11.255853699985892 seconds
- Total FRIDAY response: 11.256247600074857
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
