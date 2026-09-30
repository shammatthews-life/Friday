# FRIDAY Conversational Layer

This package defines the model-independent conversation boundary for FRIDAY. It does not load a model, access hardware, or connect to VisionAid subsystems.

## Flow

`ConversationEngine` passes the user message and compact `ConversationMemory` to an `LLM` implementation. The LLM returns either a normal answer or an internal `CapabilityRequest`. The engine resolves context references, calls a registered provider, and passes its structured `CapabilityResult` back to the LLM for a natural-language response. Only user and assistant text is returned from `process`; capability identifiers and payloads stay internal.

The `LLM` protocol is the future model adapter. No runtime or network API is implemented. Until providers are registered, capability calls return an unavailable result. A language model must not claim scene, target, depth, safety, or location facts unless those facts are present in returned provider data.

## Capability and grounding contracts

Capability identifiers and argument schemas are extensible. `CapabilityKind` distinguishes information requests from actions. Arguments can refer to compact conversation fields with a structured value such as `{"$context": "current_target"}`; unresolved references are clarified without invoking a provider.

Providers return `CapabilityResult` data rather than user-facing prose. Optional `GroundingData` slots cover scene information, target information, relative depth, safety state, and location information. Future reminders or general-assistant functions can use the same request/provider/result boundary without changing user-facing language.

## Conversation memory

Memory keeps a bounded recent message list, current topic, current referent, current task, current target, and timestamp. Internal capability requests and results are not added to the user-visible message history.

## Configuration

`configs/llm.yaml` contains tuning placeholders for FRIDAY's name, tone, verbosity, friendliness, humor, response length, and safety communication style. This module does not load that file yet; a future runtime/config integration can do so.

## Mock architecture test

Run `python scripts/test_conversation_architecture.py` from the repository root. The test uses only Python mocks and does not invoke a model, package installation, downloads, hardware, or existing perception/speech components.