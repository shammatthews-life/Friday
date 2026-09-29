from __future__ import annotations

import time
import re
from dataclasses import dataclass
from pathlib import Path

from src.assistant.intent import IntentType, ParsedIntent, parse_intent
from src.assistant.response import (
    respond_to_intent,
    respond_to_recently_seen,
    respond_to_scene_changes,
    respond_to_search,
)
from src.core.scene_memory import SceneMemory
from src.core.scene_state import SceneObject, SceneState
from src.search.object_search import ObjectSearch, TargetState


@dataclass
class ConversationContext:
    last_relevant_object_label: str | None = None
    last_relevant_position: str | None = None
    last_relevant_search_target: str | None = None
    timestamp: float | None = None
    last_response_intent: str | None = None


class FridayAssistant:
    def __init__(
        self,
        model_path: str | Path = "models/detection/yolo26n.pt",
        scene_memory: SceneMemory | None = None,
    ) -> None:
        self.name = "FRIDAY"
        self.ready = True
        self.model_path = Path(model_path)
        self.model = None
        self._object_boxes: dict[int, tuple[float, float, float, float]] = {}
        self.depth_estimator = None
        self.object_search = ObjectSearch()
        self.scene_memory = scene_memory or SceneMemory(miss_limit=1)
        self._last_memory_scene: SceneState | None = None
        self._scene_newly_noticed = []
        self._scene_disappeared = []
        self.conversation_context = ConversationContext()

    def update_scene_memory(self, scene: SceneState) -> None:
        if scene is self._last_memory_scene:
            return
        previous_disappearance_ids = {
            item.track_id for item in self.scene_memory.get_recently_disappeared_objects()
        }
        self.scene_memory.update(scene)
        self._scene_newly_noticed = self.scene_memory.get_newly_noticed_objects()
        self._scene_disappeared = [
            item
            for item in self.scene_memory.get_recently_disappeared_objects()
            if item.track_id not in previous_disappearance_ids
        ]
        self._last_memory_scene = scene

    def detect(self, image_path: str | Path) -> SceneState:
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
        timestamp = time.time()
        result = self.model.predict(source=str(image_path), device="cpu", verbose=False, imgsz=640)[0]
        height, width = result.orig_shape
        scene = SceneState(timestamp=timestamp)
        for box, confidence, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            left, top, right, bottom = [float(value) for value in box]
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            normalized_horizontal = center_x / width if width else 0.5
            position_category = (
                "left" if normalized_horizontal < 1 / 3 else
                "right" if normalized_horizontal > 2 / 3 else
                "center"
            )
            obj = SceneObject(
                label=str(result.names[int(class_id)]),
                confidence=float(confidence),
                center_x=center_x,
                center_y=center_y,
                normalized_horizontal=normalized_horizontal,
                position_category=position_category,
                timestamp=timestamp,
            )
            scene.add_object(obj)
            self._object_boxes[id(obj)] = (left, top, right, bottom)
        return scene

    def parse(self, text: str) -> ParsedIntent:
        return parse_intent(text)

    def bounding_box_for(self, obj: SceneObject) -> tuple[float, float, float, float] | None:
        return self._object_boxes.get(id(obj))

    @staticmethod
    def _normalize_context_label(label: str) -> str:
        normalized = re.sub(r"\s+", " ", label.strip().lower())
        if normalized == "phone":
            return "cell phone"
        if normalized == "people":
            return "person"
        return normalized

    @staticmethod
    def _is_context_reference(label: str | None) -> bool:
        return label is not None and label.lower().strip() in {
            "it", "that", "that object", "same one", "the same one"
        }

    def _record_context(
        self,
        scene: SceneState,
        label: str,
        intent: str,
        search_target: str | None = None,
    ) -> bool:
        normalized = self._normalize_context_label(label)
        matches = [obj for obj in scene.objects if self._normalize_context_label(obj.label) == normalized]
        self.conversation_context.timestamp = max(
            [scene.timestamp, *(obj.timestamp for obj in scene.objects)], default=0.0
        )
        self.conversation_context.last_relevant_search_target = (
            search_target or self.conversation_context.last_relevant_search_target
        )
        if len(matches) > 1:
            self.conversation_context.last_relevant_object_label = None
            self.conversation_context.last_relevant_position = None
            self.conversation_context.last_response_intent = "AMBIGUOUS"
            return False
        self.conversation_context.last_relevant_object_label = normalized
        self.conversation_context.last_relevant_position = matches[0].position_category if matches else None
        self.conversation_context.last_response_intent = intent
        return True

    def _mark_context_ambiguous(self, scene: SceneState) -> None:
        self.conversation_context.last_relevant_object_label = None
        self.conversation_context.last_relevant_position = None
        self.conversation_context.timestamp = max(
            [scene.timestamp, *(obj.timestamp for obj in scene.objects)], default=0.0
        )
        self.conversation_context.last_response_intent = "AMBIGUOUS"

    @staticmethod
    def _depth_query_label(text: str) -> str | None:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        match = re.fullmatch(r"how far is (?:a |an |the )?(.+?)[?!.]*", normalized)
        if not match:
            match = re.fullmatch(
                r"is (?:a |an |the )?(.+?) (?:near|close|far)[?!.]*", normalized
            )
        return match.group(1).strip() if match and match.group(1).strip() else None

    def relative_depth_for_object(
        self, image_path: str | Path, obj: SceneObject
    ) -> RelativeDepthResult:
        from src.perception.depth.depth_estimator import DepthEstimator, RelativeDepthResult

        box = self.bounding_box_for(obj)
        if box is None:
            return RelativeDepthResult(None, "unable to estimate")
        if self.depth_estimator is None:
            self.depth_estimator = DepthEstimator()
        depth_map = self.depth_estimator.estimate(image_path)
        return self.depth_estimator.relative_depth_for_box(depth_map, box)

    def respond(
        self, text: str, scene: SceneState, image_path: str | Path | None = None
    ) -> str:
        self.update_scene_memory(scene)
        parsed = self.parse(text)
        depth_label = self._depth_query_label(text)
        requested_label = parsed.object_label or depth_label
        contextual_followup = self._is_context_reference(requested_label)
        if contextual_followup:
            context = self.conversation_context
            if context.last_response_intent == "AMBIGUOUS":
                return "I am not sure which object you mean."
            if context.last_relevant_object_label is None:
                return "I don't know which object you mean."
            requested_label = context.last_relevant_object_label
            if depth_label is not None:
                depth_label = requested_label
            parsed = ParsedIntent(parsed.intent, requested_label)

        explicit_label = None if contextual_followup else requested_label
        if parsed.intent.name == "SCENE_CHANGES":
            return respond_to_scene_changes(self._scene_newly_noticed, self._scene_disappeared)
        if parsed.intent.name == "RECENTLY_SEEN":
            response = respond_to_recently_seen(self.scene_memory, parsed.object_label)
            if explicit_label:
                self._record_context(scene, explicit_label, parsed.intent.value)
            return response
        if parsed.intent.name == "SEARCH_OBJECT":
            search_label = self._normalize_context_label(requested_label or "")
            if not search_label:
                return "I am not sure what object you want me to find."
            matching = [
                obj for obj in scene.objects
                if self._normalize_context_label(obj.label) == search_label
            ]
            if len(matching) > 1:
                highest_confidence = max(obj.confidence for obj in matching)
                if sum(obj.confidence == highest_confidence for obj in matching) > 1:
                    self._mark_context_ambiguous(scene)
                    return "I am not sure which object you mean."
            current_target = self.object_search.target
            if (
                not contextual_followup
                and
                current_target is not None
                and current_target.label == search_label
                and current_target.state is not TargetState.LOST
            ):
                target = self.object_search.update(scene)
            else:
                target = self.object_search.begin_search(search_label, scene)
            response = respond_to_search(target)
            self._record_context(scene, search_label, parsed.intent.value, search_target=search_label)
            return response

        if depth_label is not None:
            depth_label = self._normalize_context_label(depth_label)
            matches = [obj for obj in scene.objects if obj.label.lower() == depth_label]
            if len(matches) > 1:
                self._mark_context_ambiguous(scene)
                return "I am not sure which object you mean."
            if explicit_label or contextual_followup:
                self._record_context(scene, depth_label, "RELATIVE_DEPTH_QUERY")
            if not matches:
                return f"I do not see a {depth_label}."
            if image_path is None:
                return f"I can see the {depth_label}, but I cannot estimate its relative depth."
            result = self.relative_depth_for_object(image_path, matches[0])
            if result.value is None:
                return f"I can see the {depth_label}, but I cannot estimate its relative depth."
            return f"The {depth_label} appears {result.category}."

        if parsed.intent.name == "WHAT_AROUND":
            if len(scene.objects) == 1:
                self._record_context(scene, scene.objects[0].label, parsed.intent.value)
            elif len(scene.objects) > 1:
                self._mark_context_ambiguous(scene)
            return respond_to_intent(parsed, scene, self.scene_memory)

        context_label = requested_label if contextual_followup else explicit_label
        if context_label and parsed.intent in {
            IntentType.IS_OBJECT_PRESENT,
            IntentType.WHERE_OBJECT,
            IntentType.COUNT_OBJECT,
        }:
            label = self._normalize_context_label(context_label)
            matches = [obj for obj in scene.objects if self._normalize_context_label(obj.label) == label]
            if parsed.intent is IntentType.WHERE_OBJECT and len(matches) > 1:
                self._mark_context_ambiguous(scene)
                return "I am not sure which object you mean."
            self._record_context(scene, label, parsed.intent.value)
            parsed = ParsedIntent(parsed.intent, label)

        return respond_to_intent(parsed, scene, self.scene_memory)
