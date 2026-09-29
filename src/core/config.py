from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RuntimeConfig:
    project_root: Path
    model_root: Path
    device_yaml: Path

    @classmethod
    def from_project_root(cls, root: str | Path) -> "RuntimeConfig":
        root_path = Path(root).resolve()
        return cls(
            project_root=root_path,
            model_root=root_path / "models",
            device_yaml=root_path / "configs" / "device.yaml",
        )
