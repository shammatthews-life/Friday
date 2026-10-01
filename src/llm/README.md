# FRIDAY Conversational Layer

This package defines the conversation boundary for FRIDAY. `ConversationEngine` and its capability contracts remain model-independent. `VisionAidBridge` registers the existing scene-awareness, object-search, and relative-depth capabilities with that conversation boundary. The repository's current llama.cpp Qwen smoke test is standalone and does not implement the `LLM` protocol; this bridge is ready to be registered beside a future adapter, but does not start or own the model runtime.

## Flow

`ConversationEngine` passes the user message and compact `ConversationMemory` to an `LLM` implementation. The LLM returns either a normal answer or an internal `CapabilityRequest`. The engine resolves context references, calls a registered provider, and passes its structured `CapabilityResult` back to the LLM for a natural-language response. Only user and assistant text is returned from `process`; capability identifiers and payloads stay internal. The LLM decides whether a turn needs a capability; the bridge does not parse user phrases or generate user-facing replies.

The `LLM` protocol is the model-adapter boundary. The bridge is registered through `VisionAidBridge.register(registry)`. Supply a callable for the latest `SceneState`; optionally supply `SceneMemory`, the existing `FridayAssistant`, and a callable for the current image path. The assistant's current search object is reused, and relative depth is requested only when needed through its existing lazy depth method. Without an assistant/image or a mock depth provider, relative depth is reported unavailable rather than inferred.

A language model must not claim scene, target, or relative-depth facts unless those facts are present in returned provider data. An unavailable scene must not be described as visible. Depth categories are relative, never metres or feet. Providers return structured results only; they do not make alternate perception, search, or depth implementations.

## Capability and grounding contracts

Capability identifiers and argument schemas are extensible. `CapabilityKind` distinguishes information requests from actions. Arguments can refer to compact conversation fields with a structured value such as `{"$context": "current_target"}`; unresolved references are clarified without invoking a provider.

Providers return `CapabilityResult` data rather than user-facing prose. Optional `GroundingData` slots cover scene information, target information, relative depth, safety state, and location information. Future reminders or general-assistant functions can use the same request/provider/result boundary without changing user-facing language.

## Conversation memory

Memory keeps a bounded recent message list, current topic, current referent, current task, current target, and timestamp. Internal capability requests and results are not added to the user-visible message history.

## Configuration

`configs/llm.yaml` contains FRIDAY's conversation style plus the internal VisionAid capability identifiers and resource policy. Capability identifiers are implementation details and must never appear in user-facing replies. This module does not load that file automatically; runtime configuration remains the caller's responsibility.

## Memory-aware capability loading

The Qwen3-8B Q4_K_M smoke test used approximately 4.65 GB process-tree RAM and left about 399 MB free system RAM. Do not preload YOLO, Depth Anything, Whisper, or Piper alongside Qwen. Keep the bridge and its providers lightweight and initialize specialists only when requested. The bridge does not manage llama-server or model eviction; the runtime/LLM adapter must enforce one-heavy-model-at-a-time scheduling and release a specialist before resuming Qwen inference.

## Local model smoke test

`scripts/test_local_llm.py` uses the official standalone `llama.cpp` server with a local GGUF file. It keeps a short multi-turn conversation, limits the model to a 2,048-token context for the initial run, and records timing and resource measurements in `data/evaluation/local_llm_smoke_test.json` and `.md`. The server binds to loopback only, and the test sets Hugging Face/Transformers offline flags. It does not access a camera, call capabilities, or connect the model to `ConversationEngine`.

The script expects the official Qwen3-8B Q4_K_M GGUF at `models/llm/qwen3-8b/Qwen3-8B-Q4_K_M.gguf` and the official Windows llama.cpp bundle extracted under `.venv/llama.cpp`. The GGUF is ignored by Git.

## Mock architecture tests

Run `python scripts/test_conversation_architecture.py` and `python scripts/test_llm_visionaid_bridge.py` from the repository root. The VisionAid bridge test uses a mock scene and mock relative-depth provider plus the existing pure-Python `ObjectSearch`; it does not open a camera or load PyTorch, YOLO, Depth Anything, Piper, Whisper, or Qwen. It verifies context resolution, grounded natural replies, hidden capability identifiers, and normal conversation without capability calls.