import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from flying_discs.morrison.base import MorrisonBaseCalculator
from flying_discs.morrison.constants import MorrisonConstants
from flying_discs.morrison.coordinates import MorrisonPosition3D, MorrisonTrajectory3D
from flying_discs.utils import angle_between_vectors, distance_v1_v2


@dataclass
class MorrisonLinearThrow:
    trajectory: MorrisonTrajectory3D
    constants: MorrisonConstants
    initial_position: MorrisonPosition3D
    v0: float
    angle_of_attack: float
    direction_angle: float
    deltaT: float
    target_x: Optional[float] = None
    target_y: Optional[float] = None


class MorrisonLinearCalculator:
    # pylint: disable = invalid-name,too-many-instance-attributes,too-many-locals
    def __init__(self, constants: MorrisonConstants) -> None:
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
                    MorrisonPosition3D(
                        initial_position.x,
                        initial_position.y,
                        initial_position.z,
                        base_position.vx * math.cos(direction_angle),
                        base_position.vx * math.sin(direction_angle),
                        base_position.vz,
                        base_position.ax * math.cos(direction_angle),
                        base_position.ax * math.sin(direction_angle),
                        base_position.az,
                    )
                )
                continue
            new_ax = base_position.ax * math.cos(direction_angle)
            new_ay = base_position.ax * math.sin(direction_angle)
            new_vx = linear_trajectory[i - 1].vx + new_ax
            new_vy = linear_trajectory[i - 1].vy + new_ay
            new_x = linear_trajectory[i - 1].x + new_vx * deltaT
            new_y = linear_trajectory[i - 1].y + new_vy * deltaT
            linear_trajectory.append(
                MorrisonPosition3D(
                    new_x, new_y, base_position.z, new_vx, new_vy, base_position.vz, new_ax, new_ay, base_position.az
                )
            )
        return MorrisonLinearThrow(
            MorrisonTrajectory3D(linear_trajectory),
            self.constants,
            initial_position,
            v0,
            angle_of_attack,
            direction_angle,
            deltaT,
        )

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
        v0 = 0.0
        distance_traveled = -math.inf
        height_at_target = -math.inf
        while distance_traveled < dist or height_at_target < 0:
            v0 += 0.1
            distance_traveled = -math.inf
            trajectory = self.calculate_trajectory(
                initial_position, v0, angle_of_attack, direction_angle, deltaT
            ).trajectory
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
        return MorrisonLinearThrow(
            trajectory,
            self.constants,
            initial_position,
            v0,
            angle_of_attack,
            direction_angle,
            deltaT,
            target_x,
            target_y,
        )
