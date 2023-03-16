from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MorrisonPosition2D:
    x: float
    z: float
    vx: float
    vz: float
    ax: float
    az: float


class MorrisonTrajectory2D:
    # pylint: disable=too-many-instance-attributes, invalid-name
    def __init__(self, positions: Sequence[MorrisonPosition2D]) -> None:
        self.positions = positions
        self.X = np.array([p.x for p in self.positions])
        self.Z = np.array([p.z for p in self.positions])
        self.VX = np.array([p.vx for p in self.positions])
        self.VZ = np.array([p.vz for p in self.positions])
        self.AX = np.array([p.ax for p in self.positions])
        self.AZ = np.array([p.az for p in self.positions])

    def __iter__(self) -> "MorrisonTrajectory2D":
        # pylint: disable=attribute-defined-outside-init
        self.n = 0
        return self

    def __next__(self) -> MorrisonPosition2D:
        if self.n <= len(self.positions) - 1:
            result = self.positions[self.n]
            self.n += 1
            return result
        raise StopIteration

    def __getitem__(self, i: int) -> MorrisonPosition2D:
        return self.positions[i]

    def __len__(self) -> int:
        return len(self.positions)

    def __eq__(self, o: object) -> bool:
        return self.positions == o


@dataclass
class MorrisonPosition3D:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float


class MorrisonTrajectory3D:
    # pylint: disable=too-many-instance-attributes, invalid-name

    def __init__(self, positions: Sequence[MorrisonPosition3D]) -> None:
        self.positions = positions
        self.X = np.array([p.x for p in self.positions])
        self.Y = np.array([p.y for p in self.positions])
        self.Z = np.array([p.z for p in self.positions])
        self.VX = np.array([p.vx for p in self.positions])
        self.VY = np.array([p.vy for p in self.positions])
        self.VZ = np.array([p.vz for p in self.positions])
        self.AX = np.array([p.ax for p in self.positions])
        self.AY = np.array([p.ay for p in self.positions])
        self.AZ = np.array([p.az for p in self.positions])

    def __iter__(self) -> "MorrisonTrajectory3D":
        # pylint: disable=attribute-defined-outside-init
        self.n = 0
        return self

    def __next__(self) -> MorrisonPosition3D:
        if self.n <= len(self.positions) - 1:
            result = self.positions[self.n]
            self.n += 1
            return result
        raise StopIteration

    def __getitem__(self, i: int) -> MorrisonPosition3D:
        return self.positions[i]

    def __len__(self) -> int:
        return len(self.positions)

    def __eq__(self, o: object) -> bool:
        return self.positions == o
