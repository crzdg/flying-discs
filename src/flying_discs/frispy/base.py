from dataclasses import dataclass

from frispy.disc import Disc

from flying_discs.frispy.constants import Constants
from flying_discs.frispy.coordinates import FrispyPosition, FrispyTrajectory


@dataclass
class FrispyTrhow:
    # pylint: disable=too-many-instance-attributes
    trajectory: FrispyTrajectory
    constans: Constants
    initial_position: FrispyPosition
    vx: float
    vy: float
    vz: float
    phi: float
    theta: float
    gamma: float
    dphi: float
    dtheta: float
    dgamma: float
    deltaT: float


class FrispyCalculator:
    # pylint: disable=too-many-instance-attributes
    def __init__(self, constants: Constants) -> None:
        self.constants = constants

    def calculate_trajectory(
        # pylint: disable=too-many-arguments,too-many-locals
        self,
        initial_position: FrispyPosition,
        vx0: float,
        vy0: float,
        vz0: float,
        phi: float,
        theta: float,
        gamma: float,
        dphi: float,
        dtheta: float,
        dgamma: float,
        deltaT: float,
    ) -> FrispyTrajectory:
        disc = Disc(
            x=initial_position.x,
            y=initial_position.y,
            z=initial_position.z,
            vx=vx0,
            vy=vy0,
            vz=vz0,
            phi=phi,
            theta=theta,
            gamma=gamma,
            dphi=dphi,
            dtheta=dtheta,
            dgamma=dgamma,
            area=self.constants.AREA,
            I_xx=self.constants.I_xx,
            I_zz=self.constants.I_zz,
            mass=self.constants.MASS,
            air_density=self.constants.RH0,
            g=self.constants.GRAVITY,
            model=self.constants.MODEL,
            eom_class=self.constants.EOM,
        )
        flight_time = 20
        result, _ = disc.compute_trajectory(
            flight_time=flight_time,
            n_times=int(1 / deltaT * flight_time),
        )
        positions = []
        for i, _ in enumerate(result["times"]):
            positions.append(
                FrispyPosition(
                    x=result["x"][i],
                    y=result["y"][i],
                    z=result["z"][i],
                    vx=result["vx"][i],
                    vy=result["vy"][i],
                    vz=result["vz"][i],
                    phi=result["phi"][i],
                    theta=result["theta"][i],
                    gamma=result["gamma"][i],
                    dphi=result["dphi"][i],
                    dtheta=result["dtheta"][i],
                    dgamma=result["dgamma"][i],
                )
            )
        return FrispyTrajectory(positions)
