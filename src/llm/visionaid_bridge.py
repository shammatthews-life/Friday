from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.core.scene_memory import SceneMemory
from src.core.scene_state import SceneObject, SceneState
from src.llm.tool_interface import (
    CapabilityKind,
    CapabilityProvider,
    CapabilityRegistry,
    CapabilityRequest,
    CapabilityResult,
    GroundingData,
)
from src.search.object_search import ObjectSearch, TargetState


SCENE_AWARENESS = "vision.scene_awareness"
OBJECT_SEARCH = "vision.object_search"
RELATIVE_DEPTH = "vision.relative_depth"

SceneProvider = Callable[[], SceneState | None]
ImagePathProvider = Callable[[], str | Path | None]
DepthProvider = Callable[[SceneObject, str | Path | None], Any]


class VisionAidBridge(CapabilityProvider):
    """Expose existing VisionAid data through the conversation capability API.

    Scene and image providers are queried only when a matching capability is
    requested. Depth inference is delegated to FridayAssistant's existing lazy
    relative-depth method unless a mock/custom depth provider is injected.
    """

    capabilities = (SCENE_AWARENESS, OBJECT_SEARCH, RELATIVE_DEPTH)

    def __init__(
        self,
        scene_provider: SceneProvider | None = None,
        *,
        scene_memory: SceneMemory | None = None,
        assistant: Any | None = None,
        object_search: ObjectSearch | None = None,
        image_path_provider: ImagePathProvider | None = None,
        depth_provider: DepthProvider | None = None,
    ) -> None:
        self.scene_provider = scene_provider
        self.scene_memory = scene_memory
        self.assistant = assistant
        self.object_search = object_search
        self.image_path_provider = image_path_provider
        self.depth_provider = depth_provider

    def register(self, registry: CapabilityRegistry) -> CapabilityRegistry:
        for capability in self.capabilities:
            registry.register(capability, self)
        return registry

    def provide(self, request: CapabilityRequest) -> CapabilityResult:
        if request.kind is not CapabilityKind.INFORMATION:
            return CapabilityResult.unavailable("VisionAid capabilities are information-only")
        if request.capability == SCENE_AWARENESS:
            return self._scene_awareness(request)
        if request.capability == OBJECT_SEARCH:
            return self._object_search_result(request)
        if request.capability == RELATIVE_DEPTH:
            return self._relative_depth(request)
        return CapabilityResult.unavailable("Unknown VisionAid capability")

    def _current_scene(self) -> SceneState | None:
        return self.scene_provider() if self.scene_provider is not None else None

    @staticmethod
    def _object_data(obj: Any) -> dict[str, Any]:
        return {
            "label": str(obj.label),
            "position": str(
                getattr(obj, "position_category", getattr(obj, "position", "unknown"))
            ),
            "confidence": float(obj.confidence),
        }

    @staticmethod
    def _label(request: CapabilityRequest) -> str | None:
        for key in ("target", "object", "label"):
            value = request.arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _normalize_label(label: str) -> str:
        normalized = " ".join(label.lower().strip().split())
        return ObjectSearch.LABEL_ALIASES.get(normalized, normalized)

    def _scene_awareness(self, request: CapabilityRequest) -> CapabilityResult:
        scene = self._current_scene()
        objects = [self._object_data(obj) for obj in scene.objects] if scene else []
        remembered = []
        if self.scene_memory is not None:
            remembered = [
                {
                    "label": obj.label,
                    "position": obj.position,
                    "confidence": float(obj.confidence),
                }
                for obj in self.scene_memory.get_visible_objects()
            ]

        target = self._label(request)
        target_object = None
        if target:
            normalized_target = self._normalize_label(target)
            target_object = next(
                (
                    obj
                    for obj in objects
                    if self._normalize_label(obj["label"]) == normalized_target
                ),
                None,
            )
        data = {
            "scene_available": scene is not None,
            "objects": objects,
            "recently_seen": remembered,
        }
        if target:
            data["target"] = target
            data["target_object"] = target_object
        return CapabilityResult(
            available=True,
            data=data,
            grounding=GroundingData(scene_information=data),
        )

    def _get_object_search(self) -> ObjectSearch:
        if self.object_search is not None:
            return self.object_search
        if self.assistant is not None:
            self.object_search = self.assistant.object_search
        else:
            self.object_search = ObjectSearch()
        return self.object_search

    def _object_search_result(self, request: CapabilityRequest) -> CapabilityResult:
        target = self._label(request)
        scene = self._current_scene()
        if target is None:
            return CapabilityResult.unavailable("A target label is required")
        if scene is None:
            data = {"scene_available": False, "found": False, "target": target}
            return CapabilityResult(
                available=True,
                data=data,
                grounding=GroundingData(target_information=data),
            )

        lock = self._get_object_search().begin_search(target, scene)
        found = lock.state in {TargetState.FOUND, TargetState.LOCKED}
        data = {
            "scene_available": True,
            "found": found,
            "target": lock.label,
            "position": lock.position,
            "confidence": lock.confidence,
            "state": lock.state.value,
        }
        return CapabilityResult(
            available=True,
            data=data,
            grounding=GroundingData(target_information=data),
        )

    def _find_scene_object(self, scene: SceneState, label: str) -> SceneObject | None:
        normalized = self._normalize_label(label)
        return next(
            (
                obj
                for obj in scene.objects
                if self._normalize_label(obj.label) == normalized
            ),
            None,
        )

    def _relative_depth(self, request: CapabilityRequest) -> CapabilityResult:
        target = self._label(request)
        scene = self._current_scene()
        if target is None:
            return CapabilityResult.unavailable("A target label is required")
        if scene is None:
            data = {"scene_available": False, "target": target, "depth_available": False}
            return CapabilityResult(
                available=True,
                data=data,
                grounding=GroundingData(relative_depth=data),
            )
        scene_object = self._find_scene_object(scene, target)
        if scene_object is None:
            data = {
                "scene_available": True,
                "target": target,
                "object_found": False,
                "depth_available": False,
            }
            return CapabilityResult(
                available=True,
                data=data,
                grounding=GroundingData(relative_depth=data),
            )

        image_path = self.image_path_provider() if self.image_path_provider else None
        if self.depth_provider is not None:
            depth_result = self.depth_provider(scene_object, image_path)
        elif self.assistant is not None and image_path is not None:
            depth_result = self.assistant.relative_depth_for_object(image_path, scene_object)
        else:
            data = {
                "scene_available": True,
                "target": scene_object.label,
                "object_found": True,
                "depth_available": False,
            }
            return CapabilityResult(
                available=True,
                data=data,
                grounding=GroundingData(relative_depth=data),
            )

        if isinstance(depth_result, Mapping):
            category = depth_result.get("category")
            depth_available = bool(depth_result.get("available", category is not None))
        else:
            category = getattr(depth_result, "category", None)
            depth_available = category is not None and getattr(depth_result, "value", 1) is not None
        data = {
            "scene_available": True,
            "target": scene_object.label,
            "object_found": True,
            "depth_available": depth_available,
            "category": str(category) if depth_available else None,
            "depth_type": "relative",
        }
        return CapabilityResult(
            available=True,
            data=data,
            grounding=GroundingData(relative_depth=data),
        )
