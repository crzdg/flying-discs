from typing import Any, Dict, Tuple

import diffrax
import equinox as eqx
import jax.numpy as jnp
from diffrax import Solution

from flying_discs.frispy.jax_backend.eom import EOM
from flying_discs.frispy.jax_backend.model import Model


class Disc(eqx.Module):
    # pylint: disable=too-many-instance-attributes
    """Flying spinning disc object. Diffrax/JAX implementation."""

    area: float
    I_xx: float
    I_zz: float
    mass: float
    air_density: float
    g: float
    model: Model
    eom: EOM

    def __init__(
        self,
        area: float = 0.058556,
        I_xx: float = 0.001219,
        I_zz: float = 0.002352,
        mass: float = 0.175,
        air_density: float = 1.225,
        g: float = 9.81,
    ) -> None:
        # pylint: disable=duplicate-code
        self.area = area
        self.I_xx = I_xx
        self.I_zz = I_zz
        self.mass = mass
        self.air_density = air_density
        self.g = g
        self.model = Model()
        self.eom = EOM(
            area=self.area,
            I_xx=self.I_xx,
            I_zz=self.I_zz,
            mass=self.mass,
            air_density=self.air_density,
            g=self.g,
        )

    # pylint: disable=duplicate-code
    def compute_trajectory(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 1.0,
        vx: float = 10.0,
        vy: float = 0.0,
        vz: float = 0.0,
        phi: float = 0.0,
        theta: float = 0.0,
        gamma: float = 0.0,
        dphi: float = 0.0,
        dtheta: float = 0.0,
        dgamma: float = 62.0,
        flight_time: float = 3.0,
        n_times: int = 100,
        **solver_kwargs: Any,
    ) -> Tuple[Dict[str, jnp.ndarray], Solution]:
        # pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals
        """Call the Diffrax differential equation solver to compute the trajectory."""

        t_span = solver_kwargs.pop("t_span", (0.0, flight_time))
        t_eval = solver_kwargs.pop("t_eval", jnp.linspace(t_span[0], t_span[1], n_times))

        y0 = jnp.array(
            [
                x,
                y,
                z,
                vx,
                vy,
                vz,
                phi,
                theta,
                gamma,
                dphi,
                dtheta,
                dgamma,
            ]
        )

        # ---------------------------------------------------------
        # Diffrax Solver Setup
        # ---------------------------------------------------------
        # TODO: Fix types for ODETerm, works for now
        term = diffrax.ODETerm(self.eom.compute_derivatives)  # type: ignore
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=t_eval)
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

        result = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            args=None,  # Args are optional in Diffrax, we baked parameters into `eom` PyTree
            **solver_kwargs,
        )

        # result.ys will have shape (n_times, 12). We slice it out to match the original dict.
        ys = result.ys
        return (
            {
                "times": result.ts,
                "x": ys[:, 0],
                "y": ys[:, 1],
                "z": ys[:, 2],
                "vx": ys[:, 3],
                "vy": ys[:, 4],
                "vz": ys[:, 5],
                "phi": ys[:, 6],
                "theta": ys[:, 7],
                "gamma": ys[:, 8],
                "dphi": ys[:, 9],
                "dtheta": ys[:, 10],
                "dgamma": ys[:, 11],
            },
            result,
        )
