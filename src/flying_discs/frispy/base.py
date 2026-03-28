from dataclasses import dataclass
from typing import Any, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from flying_discs.frispy.constants import Constants
from flying_discs.frispy.coordinates import FrispyPosition, FrispyTrajectory
from flying_discs.frispy.jax_backend.disc import Disc as JAXDisc
from flying_discs.frispy.scipy_backend.disc import Disc


class ODEKwargs(TypedDict):
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    phi: float
    theta: float
    gamma: float
    dphi: float
    dtheta: float
    dgamma: float
    flight_time: float
    n_times: int


@eqx.filter_jit
def run_jax_sim(d: JAXDisc, kwargs: ODEKwargs, solver_kwargs: dict[str, Any]) -> dict[str, jnp.ndarray]:
    res, _ = d.compute_trajectory(**kwargs, **solver_kwargs)
    return res


@dataclass
class FrispyThrow:
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
    def __init__(self, constants: Constants, use_jax: bool = False, use_gpu: bool = False) -> None:
        self.constants = constants
        self._use_jax = use_jax

        if use_gpu and not use_jax:
            raise ValueError("GPU acceleration is only available with JAX. Set use_jax=True to use GPU.")
        self._use_gpu = use_gpu

        if not self._use_jax:
            self._disc = Disc(
                area=constants.AREA,
                I_xx=constants.I_xx,
                I_zz=constants.I_zz,
                mass=constants.MASS,
                air_density=constants.RH0,
                g=constants.GRAVITY,
            )

        else:

            base_disc = JAXDisc(
                area=constants.AREA,
                I_xx=constants.I_xx,
                I_zz=constants.I_zz,
                mass=constants.MASS,
                air_density=constants.RH0,
                g=constants.GRAVITY,
            )

            if self._use_gpu:
                try:
                    target_device = jax.devices("gpu")[0]
                except RuntimeError as exc:
                    raise RuntimeError("GPU requested, but JAX cannot find a valid CUDA device.") from exc
            else:
                target_device = jax.devices("cpu")[0]

            # Lock the object to the chosen hardware
            self._disc = jax.device_put(base_disc, target_device)

    def calculate_trajectory(
        # pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
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

        flight_time = 20.0
        n_times = int(1 / deltaT * flight_time)

        kwargs: ODEKwargs = {
            "x": initial_position.x,
            "y": initial_position.y,
            "z": initial_position.z,
            "vx": vx0,
            "vy": vy0,
            "vz": vz0,
            "phi": phi,
            "theta": theta,
            "gamma": gamma,
            "dphi": dphi,
            "dtheta": dtheta,
            "dgamma": dgamma,
            "flight_time": flight_time,
            "n_times": n_times,
        }

        solver_kwargs: dict[str, Any] = {}

        if self._use_jax and isinstance(self._disc, JAXDisc):
            jax_result = run_jax_sim(self._disc, kwargs, solver_kwargs)
            result = {k: np.asarray(v) for k, v in jax_result.items()}
        else:
            scipy_result, _ = self._disc.compute_trajectory(**kwargs, **solver_kwargs)
            result = scipy_result

        times = np.asarray(result["times"])

        positions = []
        for i in range(len(times)):
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
