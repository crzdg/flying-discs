from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from flying_discs.morrison.constants import MorrisonConstants
from flying_discs.morrison.coordinates import MorrisonPosition3D, MorrisonTrajectory3D
from flying_discs.morrison.linear import MorrisonLinearCalculator
from flying_discs.utils import quadratic_bezier, rotate_points_around_mid_point


@dataclass
class MorrisonBezierThrowExtra:
    bezier_points: List[Tuple[float, float]]
    linear_trajectory: MorrisonTrajectory3D
    intersect: List[Tuple[float, float]]


@dataclass
class MorrisonBezierThrow:
    # pylint: disable=too-many-instance-attributes
    trajectory: MorrisonTrajectory3D
    constants: MorrisonConstants
    initial_position: MorrisonPosition3D
    v0: float
    angle_of_attack: float
    direction_angle: float
    intersect_angle: float
    factor: float
    deltaT: float
    # make an exception for the compare for the extras for easier testing
    # TODO: remove compare=False
    extras: Optional[MorrisonBezierThrowExtra] = field(default=None, compare=False)
    target_x: Optional[float] = None
    target_y: Optional[float] = None


class MorrisonBezierCalculator:
    def __init__(self, constants: MorrisonConstants) -> None:
        self._linear_calculator = MorrisonLinearCalculator(constants)
        self.constants = constants

    @staticmethod
    def _determine_bezier_points(
        trajectory: Sequence[MorrisonPosition3D],
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

    def _update_trajectory_to_bezier(
        self, trajectory: MorrisonTrajectory3D, intersect_angle: float, factor: float
    ) -> Tuple[MorrisonTrajectory3D, MorrisonBezierThrowExtra]:
        updated_positions = deepcopy(trajectory.positions)
        intersect = rotate_points_around_mid_point(
            [(p.x, p.y) for p in updated_positions],
            intersect_angle,
        )

        tmax = len(updated_positions) - 1
        clipped_factor = min(1, max(-1, (factor + 1) / 2))
        bezier_points = self._determine_bezier_points(
            updated_positions,
            intersect,
            tmax,
            clipped_factor,
        )
        for t in range(tmax):
            x, y = quadratic_bezier(
                t / tmax,
                bezier_points[0],
                bezier_points[1],
                bezier_points[2],
            )
            updated_positions[t].x = x
            updated_positions[t].y = y

        return MorrisonTrajectory3D(updated_positions), MorrisonBezierThrowExtra(bezier_points, trajectory, intersect)

    def calculate_trajectory(
        self,
        initial_position: MorrisonPosition3D,
        v0: float,
        angle_of_attack: float,
        direction_angle: float,
        intersect_angle: float,
        factor: float,
        deltaT: float,
    ) -> MorrisonBezierThrow:
        # pylint: disable=too-many-locals
        linear_throw = self._linear_calculator.calculate_trajectory(
            initial_position, v0, angle_of_attack, direction_angle, deltaT
        )
        trajectory, extras = self._update_trajectory_to_bezier(linear_throw.trajectory, intersect_angle, factor)
        return MorrisonBezierThrow(
            trajectory,
            self.constants,
            initial_position,
            v0,
            angle_of_attack,
            direction_angle,
            intersect_angle,
            factor,
            deltaT,
            extras,
        )

    def calculate_trajectory_to_position(
        self,
        initial_position: MorrisonPosition3D,
        angle_of_attack: float,
        intersect_angle: float,
        factor: float,
        target_x: float,
        target_y: float,
        deltaT: float,
    ) -> MorrisonBezierThrow:
        linear_throw = self._linear_calculator.calculate_trajectory_to_position(
            initial_position, angle_of_attack, target_x, target_y, deltaT
        )
        trajectory, extras = self._update_trajectory_to_bezier(linear_throw.trajectory, intersect_angle, factor)
        return MorrisonBezierThrow(
            trajectory,
            self.constants,
            initial_position,
            linear_throw.v0,
            angle_of_attack,
            linear_throw.direction_angle,
            intersect_angle,
            factor,
            deltaT,
            extras,
            target_x,
            target_y,
        )
