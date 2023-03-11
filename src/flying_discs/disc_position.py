from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiscPosition:
    # pylint: disable = invalid-name,too-many-instance-attributes
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    def ground(self) -> None:
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0

    def reset(self) -> None:
        self.ground()
        self.x = 0.0
        self.y = 0.0

    @classmethod
    def position(cls, **kwargs: Any) -> DiscPosition:
        return cls(**kwargs)
