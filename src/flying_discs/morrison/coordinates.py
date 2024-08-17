from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MorrisonPosition2D:
    def __init__(
        self,
        *,
        x: float = 0.0,
        z: float = 0.0,
        vx: float = 0.0,
        vz: float = 0.0,
        ax: float = 0.0,
        az: float = 0.0,
    ):
        self.x = x
        self.z = z
        self.vx = vx
        self.vz = vz
        self.ax = ax
        self.az = az


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
        self.__n = 0  # iterator count

    def __iter__(self) -> "MorrisonTrajectory2D":
        self.__n = 0
        return self

    def __next__(self) -> MorrisonPosition2D:
        if self.__n <= len(self.positions) - 1:
            result = self.positions[self.__n]
            self.__n += 1
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
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        ax: float = 0.0,
        ay: float = 0.0,
        az: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.ax = ax
        self.ay = ay
        self.az = az


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
        self.__n = 0  # iterator count

    def __iter__(self) -> "MorrisonTrajectory3D":
        self.__n = 0
        return self

    def __next__(self) -> MorrisonPosition3D:
        if self.__n <= len(self.positions) - 1:
            result = self.positions[self.__n]
            self.__n += 1
            return result
        raise StopIteration

    def __getitem__(self, i: int) -> MorrisonPosition3D:
        return self.positions[i]

    def __len__(self) -> int:
        return len(self.positions)

    def __eq__(self, o: object) -> bool:
        return self.positions == o
