# FRIDAY Conversational Layer

This package defines FRIDAY's model-independent conversation boundary. `QwenAdapter` implements the existing `LLM` protocol over llama.cpp's local OpenAI-compatible HTTP endpoint, and `VisionAidBridge` registers the existing scene-awareness, object-search, and relative-depth capabilities. `ConversationEngine` routes internal structured requests to providers and returns only natural assistant text.

## Flow

`ConversationEngine` passes the user message and compact `ConversationMemory` to an `LLM` implementation. The LLM returns either a normal answer or an internal `CapabilityRequest`. The engine resolves context references, calls a registered provider, and passes its structured `CapabilityResult` back to the LLM for a natural-language response. Only user and assistant text is returned from `process`; capability identifiers and payloads stay internal. The LLM decides whether a turn needs a capability; the bridge does not parse user phrases or generate user-facing replies.

`QwenAdapter` implements `decide()` and `respond_to_capability()`. Decisions use constrained JSON internally; unsupported capabilities are rejected and internal names are never returned by the conversation engine. Register the bridge through `VisionAidBridge.register(registry)`. Supply a callable for the latest `SceneState`; optionally supply `SceneMemory`, the existing `FridayAssistant`, and a callable for the current image path. The assistant's current search object is reused, and relative depth is requested only when needed through its existing lazy depth method. Without an assistant/image or a mock depth provider, relative depth is reported unavailable rather than inferred.

A language model must not claim scene, target, or relative-depth facts unless those facts are present in returned provider data. An unavailable scene must not be described as visible. Depth categories are relative, never metres or feet. Providers return structured results only; they do not make alternate perception, search, or depth implementations.

## Capability and grounding contracts

Capability identifiers and argument schemas are extensible. `CapabilityKind` distinguishes information requests from actions. Arguments can refer to compact conversation fields with a structured value such as `{"$context": "current_target"}`; unresolved references are clarified without invoking a provider.

Providers return `CapabilityResult` data rather than user-facing prose. Optional `GroundingData` slots cover scene information, target information, relative depth, safety state, and location information. Future reminders or general-assistant functions can use the same request/provider/result boundary without changing user-facing language.

## Conversation memory

Memory keeps a bounded recent message list, current topic, current referent, current task, current target, and timestamp. Internal capability requests and results are not added to the user-visible message history.

## Configuration

`configs/llm.yaml` contains FRIDAY's conversation style, local Qwen server defaults, internal VisionAid capability identifiers, grounding rules, and resource policy. Capability identifiers are implementation details and must never appear in user-facing replies. The adapter currently uses matching safe defaults in Python; callers may override model/runtime paths and server settings through its constructor.

## Memory-aware capability loading

The Qwen3-8B Q4_K_M smoke test used approximately 4.65 GB process-tree RAM and left about 399 MB free system RAM. Do not preload YOLO, Depth Anything, Whisper, or Piper alongside Qwen. Keep the bridge and its providers lightweight and initialize specialists only when requested. `QwenAdapter` starts only the local GGUF path, binds to `127.0.0.1`, explicitly selects the detected NVIDIA RTX 4050 Vulkan device, and terminates only a server process it owns. It can also connect to a healthy Qwen server on its configured localhost endpoint. The bridge does not preload perception models; the application must keep heavyweight specialist inference sequential and release specialist resources before Qwen resumes.

## Local model smoke test

`scripts/test_local_llm.py` is the original standalone smoke test. `scripts/test_qwen_friday.py` exercises the real Qwen adapter through `ConversationEngine`, uses only mock VisionAid providers, and writes `data/evaluation/qwen_friday_integration.json` and `.md`. It does not access a camera, microphone, Whisper, or real perception models.

The script expects the official Qwen3-8B Q4_K_M GGUF at `models/llm/qwen3-8b/Qwen3-8B-Q4_K_M.gguf` and the official Windows llama.cpp bundle extracted under `.venv/llama.cpp`. The GGUF is ignored by Git.

## Mock architecture tests

Run `python scripts/test_conversation_architecture.py` and `python scripts/test_llm_visionaid_bridge.py` for mock-only tests. Run `python scripts/test_qwen_friday.py` for the local Qwen integration test; it starts the existing local llama.cpp runtime and uses mock VisionAid providers, not live hardware or specialist models.