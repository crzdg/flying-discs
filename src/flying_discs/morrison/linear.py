import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from flying_discs.morrison.base import MorrisonBaseCalculator
from flying_discs.morrison.constants import Constants
from flying_discs.morrison.coordinates import (
    MorrisonPosition2D,
    MorrisonPosition3D,
    MorrisonTrajectory2D,
    MorrisonTrajectory3D,
)
from flying_discs.utils import angle_between_vectors, distance_v1_v2


@dataclass
class MorrisonLinearThrow:
    # pylint: disable=too-many-instance-attributes
    trajectory: MorrisonTrajectory3D
    base_trajectory: MorrisonTrajectory2D = field(compare=False)
    constants: Constants
    initial_position: MorrisonPosition3D
    v0: float
    angle_of_attack: float
    direction_angle: float
    deltaT: float
    target_x: Optional[float] = None
    target_y: Optional[float] = None


class MorrisonLinearCalculator:
    def __init__(self, constants: Constants) -> None:
        self._base_calculator = MorrisonBaseCalculator(constants)
        self.constants = constants

    def calculate_trajectory(
        self,
        initial_position: MorrisonPosition3D,
        v0: float,
        angle_of_attack: float,
        direction_angle: float,
        deltaT: float,
    ) -> MorrisonLinearThrow:
        base_trajectory = self._base_calculator.calculate_trajectory(
            initial_position.z, v0, angle_of_attack, deltaT
        ).trajectory
        linear_trajectory: List[MorrisonPosition3D] = []
        for i, base_position in enumerate(base_trajectory):
            if i == 0:
                linear_trajectory.append(
                    self._calcualte_initial_trajectory_step(initial_position, base_position, direction_angle)
                )
                continue
            linear_trajectory.append(
                self._calculate_next_trajectory_step(base_position, linear_trajectory[i - 1], direction_angle, deltaT)
            )
        return MorrisonLinearThrow(
            MorrisonTrajectory3D(linear_trajectory),
            base_trajectory,
            self.constants,
            initial_position,
            v0,
            angle_of_attack,
            direction_angle,
            deltaT,
        )

    @staticmethod
    def _calcualte_initial_trajectory_step(
        initial_position: MorrisonPosition3D, base_position: MorrisonPosition2D, direction_angle: float
    ) -> MorrisonPosition3D:
        return MorrisonPosition3D(
            x=initial_position.x,
            y=initial_position.y,
            z=initial_position.z,
            vx=base_position.vx * math.cos(direction_angle),
            vy=base_position.vx * math.sin(direction_angle),
            vz=base_position.vz,
            ax=base_position.ax * math.cos(direction_angle),
            ay=base_position.ax * math.sin(direction_angle),
            az=base_position.az,
        )

    @staticmethod
    def _calculate_next_trajectory_step(
        base_position: MorrisonPosition2D, previous_position: MorrisonPosition3D, direction_angle: float, deltaT: float
    ) -> MorrisonPosition3D:
        new_ax = base_position.ax * math.cos(direction_angle)
        new_ay = base_position.ax * math.sin(direction_angle)
        new_vx = previous_position.vx + new_ax
        new_vy = previous_position.vy + new_ay
        new_x = previous_position.x + new_vx * deltaT
        new_y = previous_position.y + new_vy * deltaT
        return MorrisonPosition3D(
            x=new_x,
            y=new_y,
            z=base_position.z,
            vx=new_vx,
            vy=new_vy,
            vz=base_position.vz,
            ax=new_ax,
            ay=new_ay,
            az=base_position.az,
        )

    def _find_trajectory_to_position(
        self,
        initial_position: MorrisonPosition3D,
        angle_of_attack: float,
        direction_angle: float,
        dist: float,
        deltaT: float,
    ) -> tuple[float, MorrisonTrajectory3D, MorrisonTrajectory2D]:
        v0 = 0.0
        distance_traveled = -math.inf
        height_at_target = -math.inf
        while distance_traveled < dist or height_at_target < 0:
            # TODO: Add exit condition if no solution is found after a certain number of iterations
            v0 += 0.1
            distance_traveled = -math.inf
            base_throw = self.calculate_trajectory(initial_position, v0, angle_of_attack, direction_angle, deltaT)
            trajectory = base_throw.trajectory
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
        return v0, trajectory, base_throw.base_trajectory

    def calculate_trajectory_to_position(
        self,
        initial_position: MorrisonPosition3D,
        angle_of_attack: float,
        target_x: float,
        target_y: float,
        deltaT: float,
    ) -> MorrisonLinearThrow:
        direction_angle = angle_between_vectors((1, 0), (target_x - initial_position.x, target_y - initial_position.y))
        dist = distance_v1_v2(target_x, target_y, initial_position.x, initial_position.y)
        found_v0, trajectory, base_trajectory = self._find_trajectory_to_position(
            initial_position, angle_of_attack, direction_angle, dist, deltaT
        )
        return MorrisonLinearThrow(
            trajectory,
            base_trajectory,
            self.constants,
            initial_position,
            found_v0,
            angle_of_attack,
            direction_angle,
            deltaT,
            target_x,
            target_y,
        )
