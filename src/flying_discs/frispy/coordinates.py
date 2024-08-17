from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FrispyPosition:
    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        phi: float = 0.0,
        theta: float = 0.0,
        gamma: float = 0.0,
        dphi: float = 0.0,
        dtheta: float = 0.0,
        dgamma: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.phi = phi
        self.theta = theta
        self.gamma = gamma
        self.dphi = dphi
        self.dtheta = dtheta
        self.dgamma = dgamma


class FrispyTrajectory:
    # pylint: disable=too-many-instance-attributes, invalid-name
    def __init__(self, positions: Sequence[FrispyPosition]) -> None:
        self.positions = positions
        self.X = np.array([p.x for p in self.positions])
        self.Y = np.array([p.y for p in self.positions])
        self.Z = np.array([p.z for p in self.positions])
        self.VX = np.array([p.vx for p in self.positions])
        self.VY = np.array([p.vy for p in self.positions])
        self.VZ = np.array([p.vz for p in self.positions])
        self.PHI = np.array([p.phi for p in self.positions])
        self.THETA = np.array([p.theta for p in self.positions])
        self.GAMMA = np.array([p.gamma for p in self.positions])
        self.DPHI = np.array([p.dphi for p in self.positions])
        self.DTHETA = np.array([p.dtheta for p in self.positions])
        self.DGAMMA = np.array([p.dgamma for p in self.positions])
        self.__n = 0  # iterator count

    def __iter__(self) -> "FrispyTrajectory":
        self.__n = 0
        return self

    def __next__(self) -> FrispyPosition:
        if self.__n <= len(self.positions) - 1:
            result = self.positions[self.__n]
            self.__n += 1
            return result
        raise StopIteration

    def __getitem__(self, i: int) -> FrispyPosition:
        return self.positions[i]

    def __len__(self) -> int:
        return len(self.positions)

    def __eq__(self, o: object) -> bool:
        return self.positions == o
