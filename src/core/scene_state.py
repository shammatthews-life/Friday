from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SceneObject:
    label: str
    confidence: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    normalized_horizontal: float = 0.5
    position_category: str = "center"
    timestamp: float = 0.0
    distance: str = "unknown"


@dataclass
class SceneState:
    objects: list[SceneObject] = field(default_factory=list)
    timestamp: float = 0.0

    def add_object(self, obj: SceneObject) -> None:
        self.objects.append(obj)

    def labels(self) -> list[str]:
        return [obj.label for obj in self.objects]
