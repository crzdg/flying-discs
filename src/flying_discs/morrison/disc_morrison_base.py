import math
from typing import Any, Optional, Sequence

from flying_discs.disc import Disc
from flying_discs.morrison.morrison_constants import DiscMorrisonConstants
from flying_discs.morrison.morrison_position import DiscMorrisonPosition


class DiscMorrison(Disc):
    # pylint: disable = invalid-name,too-many-instance-attributes,too-many-locals
    def __init__(self, constants: DiscMorrisonConstants, x: float = 0.0, y: float = 0.0, z: float = 1.7) -> None:
        super().__init__(DiscMorrisonPosition(x=x, y=y, z=z))
        self.constants = constants
        self.throw_direction = 0.0
        self.throw_power = 0.0
        self.throw_aoa = 0.0
        self._aoa = 2.5

    def _calculate_trajectory_step(
        self, deltaT: float, d: float, z: float, vd: float, vz: float, CD: float, CL: float
    ) -> DiscMorrisonPosition:
        ad = -self.constants.RHO * math.pow(vd, 2) * self.constants.A * CD * deltaT
        az = (
            self.constants.RHO * math.pow(vd, 2) * self.constants.A * CL / 2 / self.constants.MASS
            + self.constants.GRAVITY
        ) * deltaT
        vd = vd + ad
        vz = vz + az
        d = d + vd * deltaT
        z = z + vz * deltaT

        return DiscMorrisonPosition(0.0, d, z, 0.0, vd, vz, 0.0, ad, az, d, vd, ad)

    def _calculate_trajectory(
        self, v0: float, alpha: float, deltaT: float, **_: Optional[Any]
    ) -> Sequence[DiscMorrisonPosition]:
        # In code of Morrision, only 2D is calculated, distance x and height y
        # In morrison y = z in here
        # In morrison x = d in here
        # d = distance
        vd = v0
        vz = 0.0
        d = 0.0
        z = self.z
        # We simulate approx. the function for x, y with a linear function towards the target
        CL = self.constants.CL(alpha=alpha)
        CD = self.constants.CD(alpha=alpha)
        trajectory = [DiscMorrisonPosition(0.0, d, z, 0.0, vd, vz, 0.0, 0.0, 0.0, d, vd, 0.0)]
        while trajectory[-1].z > 0:
            current_step = trajectory[-1]
            next_position = self._calculate_trajectory_step(
                deltaT, current_step.d, current_step.z, current_step.vd, current_step.vz, CD, CL
            )
            trajectory.append(next_position)
        return trajectory

    def calculate_trajectory(self, timescale: float, **kwargs: Any) -> None:
        self.throw_aoa = kwargs["alpha"]
        self.throw_power = kwargs["power"]
        self.trajectory = self._calculate_trajectory(self.throw_power, self.throw_aoa, timescale)

    def calculate_trajectory_to_position(self, x: float, y: float, timescale: float, **kwargs: Any) -> None:
        raise NotImplementedError("This function is not implemented.")
