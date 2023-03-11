import math
from typing import Any, Sequence, Tuple

import numpy as np

from flying_discs.disc_position import DiscPosition
from flying_discs.morrison.disc_morrison_base import DiscMorrison
from flying_discs.morrison.morrison_constants import DiscMorrisonConstants
from flying_discs.utils import angle_between_vectors, distance_v1_v2


class DiscMorrisonLinear(DiscMorrison):
    # pylint: disable = invalid-name,too-many-instance-attributes,too-many-locals
    def __init__(self, constants: DiscMorrisonConstants, x: float = 0.0, y: float = 0.0, z: float = 1.7) -> None:
        super().__init__(constants, x, y, z)
        self.constants = constants
        self.throw_direction = 0.0
        self.throw_power = 0.0
        self.throw_aoa = 0.0
        self._aoa = 2.5

    def _calculate_trajectory(self, v0: float, alpha: float, deltaT: float, **kwargs: Any) -> Sequence[DiscPosition]:
        trajectory = super()._calculate_trajectory(v0, alpha, deltaT)
        angle = kwargs["angle"]
        point: DiscPosition
        for i, point in enumerate(trajectory):
            if i == 0:
                new_vx = point.vy * math.cos(angle)
                new_vy = point.vy * math.sin(angle)
                point.vx = new_vx
                point.vy = new_vy
                point.x = self.x
                point.y = self.y
                continue
            new_ax = point.ay * math.cos(angle)
            new_ay = point.ay * math.sin(angle)
            point.ax = new_ax
            point.ay = new_ay
            point.vx = trajectory[i - 1].vx + point.ax
            point.vy = trajectory[i - 1].vy + point.ay
            point.x = trajectory[i - 1].x + point.vx * deltaT
            point.y = trajectory[i - 1].y + point.vy * deltaT
        return trajectory

    def calculate_trajectory(self, timescale: float, **kwargs: Any) -> Sequence[DiscPosition]:
        self.throw_direction = kwargs["direction"]
        self.throw_aoa = kwargs["alpha"]
        self.throw_power = kwargs["v0"]
        self.trajectory = self._calculate_trajectory(
            self.throw_power, self.throw_aoa, timescale, angle=self.throw_direction
        )
        return self.trajectory

    def calculate_trajectory_to_position(
        self, x: float, y: float, timescale: float, **kwargs: Any
    ) -> Sequence[DiscPosition]:
        self.throw_direction = angle_between_vectors((1, 0), (x - self.x, y - self.y))
        (
            self.trajectory,
            self.throw_power,
            self.throw_aoa,
        ) = self._approximate_trajectory_to_target_x_y(x, y, timescale, angle=self.throw_direction, **kwargs)
        return self.trajectory

    def _approximate_trajectory_to_target_x_y(
        self, x: float, y: float, timescale: float, **kwargs: Any
    ) -> Tuple[Sequence[DiscPosition], float, float]:
        dist = distance_v1_v2(x, y, self.x, self.y)
        v0 = 0.0
        distance_traveled = -math.inf
        height_at_target = -math.inf
        while distance_traveled < dist or height_at_target < 0:
            v0 += 0.1
            distance_traveled = -math.inf
            trajectory = self._calculate_trajectory(v0, self._aoa, timescale, **kwargs)
            distance_traveled = float(
                np.linalg.norm(
                    [trajectory[-1].x, trajectory[-1].y],
                )
            )
            d_ = np.array([np.linalg.norm([p.x, p.y]) for p in trajectory])
            idx = np.where(d_ > dist)[0]
            if idx.size > 0:
                start_idx = idx[0]
                height_at_target = trajectory[start_idx].z
        return trajectory, v0, self._aoa
