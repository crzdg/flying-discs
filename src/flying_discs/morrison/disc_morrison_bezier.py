from copy import deepcopy
from typing import Any, List, Sequence, Tuple

import numpy as np

from flying_discs.disc_position import DiscPosition
from flying_discs.morrison.disc_morrison_linear import DiscMorrisonLinear
from flying_discs.morrison.morrison_constants import DiscMorrisonConstants
from flying_discs.utils import quadratic_bezier, rotate_points_around_mid_point


class DiscMorrisonBezier(DiscMorrisonLinear):
    def __init__(self, constants: DiscMorrisonConstants, x: float = 0.0, y: float = 0.0, z: float = 1.7) -> None:
        # pylint : disable=too-many-instance-attributes
        super().__init__(constants, x, y, z)
        self._bezier_points: List[Tuple[float, float]] = []
        self._linear_trajectory: Sequence[DiscPosition] = []
        self._rotated_points: List[Tuple[float, float]] = []
        self.rotation_angle: float = 0.0
        self.factor: float = 0.0

    @staticmethod
    def _determine_bezier_points(
        trajectory: Sequence[DiscPosition],
        rotated_points: List[Tuple[float, float]],
        tmax: int,
        factor: float,
    ) -> List[Tuple[float, float]]:
        x0 = trajectory[0].x
        y0 = trajectory[0].y
        idx = np.abs(int(tmax * factor))
        x1 = rotated_points[idx][0]
        y1 = rotated_points[idx][1]
        x2 = trajectory[-1].x
        y2 = trajectory[-1].y
        bezier_points = [(x0, y0), (x1, y1), (x2, y2)]
        return bezier_points

    def _calculate_trajectory(self, v0: float, alpha: float, deltaT: float, **kwargs: Any) -> Sequence[DiscPosition]:
        trajectory = super()._calculate_trajectory(v0, alpha, deltaT, angle=kwargs["angle"])
        self._linear_trajectory = deepcopy(trajectory)
        rotated_points = rotate_points_around_mid_point(
            [(p.x, p.y) for p in trajectory],
            kwargs["rotation_angle"],
        )
        self._rotated_points = rotated_points

        tmax = len(trajectory) - 1
        factor = (kwargs["factor"] + 1) / 2
        self._bezier_points = self._determine_bezier_points(
            trajectory,
            rotated_points,
            tmax,
            factor,
        )
        for t in range(tmax):
            x, y = quadratic_bezier(
                t / tmax,
                self._bezier_points[0],
                self._bezier_points[1],
                self._bezier_points[2],
            )
            trajectory[t].x = x
            trajectory[t].y = y

        return trajectory

    def calculate_trajectory(self, timescale: float, **kwargs: Any) -> Sequence[DiscPosition]:
        self.throw_direction = kwargs["direction"]
        self.throw_aoa = kwargs["alpha"]
        self.throw_power = kwargs["power"]
        self.rotation_angle = kwargs["rotation_angle"]
        self.factor = kwargs["factor"]
        self.trajectory = self._calculate_trajectory(
            self.throw_power,
            self.throw_aoa,
            timescale,
            angle=self.throw_direction,
            rotation_angle=self.rotation_angle,
            factor=self.factor,
        )
        return self.trajectory
