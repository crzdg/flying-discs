from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from flying_discs.frispy.jax_backend.model import Model


class _GeometricQuantitiesResult(eqx.Module):
    xhat: jnp.ndarray
    yhat: jnp.ndarray
    zhat: jnp.ndarray
    angle_of_attack: jnp.ndarray
    rotation_matrix: jnp.ndarray
    w_prime: jnp.ndarray
    w_lab: jnp.ndarray
    w: jnp.ndarray


class _ForcesResult(eqx.Module):
    lift: jnp.ndarray
    drag: jnp.ndarray
    grav: jnp.ndarray
    mass: float

    @property
    def total(self) -> jnp.ndarray:
        return self.lift + self.drag + self.grav

    @property
    def acc(self) -> jnp.ndarray:
        return self.total / self.mass


class _TorquesResult(eqx.Module):
    x_lab: jnp.ndarray
    y_lab: jnp.ndarray
    z: jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray

    @property
    def total(self) -> jnp.ndarray:
        return self.x + self.y + self.z


# -------------------------------------------------------------------
# Equations of Motion (EOM)
# -------------------------------------------------------------------


class EOM(eqx.Module):
    # pylint: disable=too-many-instance-attributes
    """Equations of motion for a disc (JAX compatible)."""

    area: float
    diameter: jnp.ndarray
    I_xx: float
    I_zz: float
    mass: float
    air_density: float
    g: float
    model: Model
    force_per_v2: float
    torque_per_v2: jnp.ndarray
    grav: jnp.ndarray
    z_hat: jnp.ndarray

    def __init__(
        self,
        area: float,
        I_xx: float,
        I_zz: float,
        mass: float,
        air_density: float = 1.225,
        g: float = 9.81,
        model: Model = Model(),
    ):
        self.area = area
        self.diameter = 2 * jnp.sqrt(self.area / jnp.pi)
        self.I_xx = I_xx
        self.I_zz = I_zz
        self.mass = mass
        self.model = model
        self.air_density = air_density
        self.g = g
        self.force_per_v2 = 0.5 * self.air_density * self.area
        self.torque_per_v2 = self.force_per_v2 * self.diameter
        self.grav = jnp.array([0.0, 0.0, -self.mass * self.g])
        self.z_hat = jnp.array([0.0, 0.0, 1.0])

    @classmethod
    def rotation_matrix_from_phi_theta(cls, phi: float, theta: float) -> jnp.ndarray:
        sp, cp = jnp.sin(phi), jnp.cos(phi)
        st, ct = jnp.sin(theta), jnp.cos(theta)
        return cls.rotation_matrix(sp, cp, st, ct)

    @staticmethod
    def rotation_matrix(sp: jnp.ndarray, cp: jnp.ndarray, st: jnp.ndarray, ct: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[ct, sp * st, -st * cp], [0.0, cp, sp], [st, -sp * ct, cp * ct]])

    @classmethod
    def compute_angle_of_attack(
        cls,
        phi: jnp.ndarray,
        theta: jnp.ndarray,
        velocity: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sp, cp = jnp.sin(phi), jnp.cos(phi)
        st, ct = jnp.sin(theta), jnp.cos(theta)
        rotation_matrix = cls.rotation_matrix(sp, cp, st, ct)
        zhat = rotation_matrix[2]
        v_dot_zhat = velocity @ zhat
        v_in_plane = velocity - zhat * v_dot_zhat
        angle_of_attack = -jnp.arctan(v_dot_zhat / jnp.linalg.norm(v_in_plane))
        # pylint: disable=duplicate-code
        return (
            angle_of_attack,
            sp,
            cp,
            st,
            ct,
            rotation_matrix,
            v_dot_zhat,
            v_in_plane,
        )

    def geometric_quantities(
        self,
        phi: jnp.ndarray,
        theta: jnp.ndarray,
        velocity: jnp.ndarray,
        angular_velocity: jnp.ndarray,
    ) -> _GeometricQuantitiesResult:
        # pylint: disable=too-many-locals
        (
            angle_of_attack,
            _,
            _,
            st,
            ct,
            rotation_matrix,
            _,
            v_in_plane,
        ) = self.compute_angle_of_attack(phi, theta, velocity)

        zhat: jnp.ndarray = rotation_matrix[2]
        xhat: jnp.ndarray = v_in_plane / jnp.linalg.norm(v_in_plane)
        yhat = jnp.cross(zhat, xhat)
        w_prime = jnp.array(
            [
                angular_velocity[0] * ct,
                angular_velocity[1],
                angular_velocity[0] * st + angular_velocity[2],
            ]
        )
        w_lab = w_prime @ rotation_matrix
        w = jnp.array([xhat, yhat, zhat]) @ w_lab
        # pylint: disable=duplicate-code
        return _GeometricQuantitiesResult(
            xhat=xhat,
            yhat=yhat,
            zhat=zhat,
            angle_of_attack=angle_of_attack,
            rotation_matrix=rotation_matrix,
            w_prime=w_prime,
            w_lab=w_lab,
            w=w,
        )

    def compute_forces(
        self,
        velocity: jnp.ndarray,
        geometric_quantities: _GeometricQuantitiesResult,
    ) -> _ForcesResult:
        vhat = velocity / jnp.linalg.norm(velocity)
        force_amplitude = self.force_per_v2 * (velocity @ velocity)
        res = _ForcesResult(
            lift=self.model.lift(geometric_quantities.angle_of_attack)
            * force_amplitude
            * jnp.cross(vhat, geometric_quantities.yhat),
            drag=self.model.drag(geometric_quantities.angle_of_attack) * force_amplitude * (-vhat),
            grav=self.grav,
            mass=self.mass,
        )
        return res

    def compute_torques(
        self,
        velocity: jnp.ndarray,
        geometric_quantities: _GeometricQuantitiesResult,
    ) -> _TorquesResult:
        torque_amplitude = self.torque_per_v2 * (velocity @ velocity)
        wx, wy, wz = geometric_quantities.w
        t_x_lab = self.model.x(wx, wz) * torque_amplitude * geometric_quantities.xhat
        t_y_lab = self.model.y(geometric_quantities.angle_of_attack, wy) * torque_amplitude * geometric_quantities.yhat
        z = self.model.z(wz) * torque_amplitude * self.z_hat
        x = geometric_quantities.rotation_matrix @ t_x_lab
        y = geometric_quantities.rotation_matrix @ t_y_lab
        return _TorquesResult(x_lab=t_x_lab, y_lab=t_y_lab, z=z, x=x, y=y)

    def compute_derivatives(self, _: jnp.number, coordinates: jnp.ndarray, __: Any) -> jnp.ndarray:
        """Right hand side of the ordinary differential equations (Diffrax compatible)."""

        def calculate_physics(coords: jnp.ndarray) -> jnp.ndarray:
            velocity = coords[3:6]
            ang_velocity = coords[9:12]
            geometric_quantities = self.geometric_quantities(coords[6], coords[7], velocity, ang_velocity)
            forces = self.compute_forces(velocity, geometric_quantities)
            torques = self.compute_torques(velocity, geometric_quantities)
            return jnp.concatenate((velocity, forces.acc, ang_velocity, torques.total))

        def frozen_state(coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.zeros_like(coords)

        # Replaces `if coordinates[2] <= 0:` for JAX tracing compatibility
        y0: jnp.ndarray = jax.lax.cond(coordinates[2] <= 0, frozen_state, calculate_physics, coordinates)
        return y0
