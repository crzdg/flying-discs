from __future__ import annotations

import math
from dataclasses import dataclass

from flying_discs.morrison.constants import MorrisonConstants
from flying_discs.morrison.coordinates import MorrisonPosition2D, MorrisonTrajectory2D


@dataclass
class MorrisonBaseThrow:
    trajectory: MorrisonTrajectory2D
    constants: MorrisonConstants
    z0: float
    v0: float
    angle_of_attack: float
    deltaT: float


class MorrisonBaseCalculator:
    # pylint: disable = invalid-name,too-many-instance-attributes,too-many-locals
    def __init__(self, constants: MorrisonConstants) -> None:
        self.constants = constants

    def calculate_trajectory_step(
        self, x: float, z: float, vx: float, vz: float, CD: float, CL: float, deltaT: float
    ) -> MorrisonPosition2D:
        ax = -self.constants.RHO * math.pow(vx, 2) * self.constants.A * CD * deltaT
        az = (
            self.constants.RHO * math.pow(vx, 2) * self.constants.A * CL / 2 / self.constants.MASS
            + self.constants.GRAVITY
        ) * deltaT
        next_vx = vx + ax
        next_vz = vz + az
        next_x = x + next_vx * deltaT
        next_z = z + next_vz * deltaT
        return MorrisonPosition2D(next_x, next_z, next_vx, next_vz, ax, az)

    def calculate_trajectory(
        self,
        z0: float,
        v0: float,
        angle_of_attack: float,
        deltaT: float,
    ) -> MorrisonBaseThrow:
        CL = self.constants.CL(alpha=angle_of_attack)
        CD = self.constants.CD(alpha=angle_of_attack)
        trajectory = [MorrisonPosition2D(0.0, z0, v0, 0.0, 0.0, 0.0)]
        while trajectory[-1].z > 0:
            current_step = trajectory[-1]
            next_position = self.calculate_trajectory_step(
                current_step.x, current_step.z, current_step.vx, current_step.vz, CD, CL, deltaT
            )
            trajectory.append(next_position)
        return MorrisonBaseThrow(
            MorrisonTrajectory2D(trajectory),
            self.constants,
            z0,
            v0,
            angle_of_attack,
            deltaT,
        )
