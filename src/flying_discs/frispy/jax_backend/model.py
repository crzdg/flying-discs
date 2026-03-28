import equinox as eqx
import jax.numpy as jnp


class Model(eqx.Module):
    # pylint: disable=too-many-instance-attributes
    """Kinematic model for a disc."""

    PL0: float = 0.33
    PLa: float = 1.9
    PD0: float = 0.18
    PDa: float = 0.69
    PTxwx: float = -0.013
    PTxwz: float = -0.0017
    PTy0: float = -0.082
    PTya: float = 0.43
    PTywy: float = -0.014
    PTzwz: float = -0.000034
    alpha_0: float = 4 * jnp.pi / 180

    def lift(self, alpha: jnp.ndarray) -> jnp.ndarray:
        return self.PL0 + self.PLa * alpha

    def drag(self, alpha: jnp.ndarray) -> jnp.ndarray:
        return self.PD0 + self.PDa * (alpha - self.alpha_0) ** 2

    def x(self, wx: float, wz: float) -> float:
        return self.PTxwx * wx + self.PTxwz * wz

    def y(self, alpha: jnp.ndarray, wy: float) -> jnp.ndarray:
        return self.PTy0 + self.PTywy * wy + self.PTya * alpha

    def z(self, wz: float) -> float:
        return self.PTzwz * wz
