from dataclasses import dataclass

import numpy as np

from flying_discs.frispy.scipy_backend.model import Model


@dataclass
class _GeometricQuantitiesResult:
    xhat: np.ndarray
    yhat: np.ndarray
    zhat: np.ndarray
    angle_of_attack: float
    rotation_matrix: np.ndarray
    w_prime: np.ndarray
    w_lab: np.ndarray
    w: np.ndarray


@dataclass
class _ForcesResult:
    lift: np.ndarray
    drag: np.ndarray
    grav: np.ndarray
    mass: float

    @property
    def total(self) -> np.ndarray:
        total: np.ndarray = self.lift + self.drag + self.grav
        return total

    @property
    def acc(self) -> np.ndarray:
        acc: np.ndarray = self.total / self.mass
        return acc


@dataclass
class _TorquesResult:
    x_lab: np.ndarray
    y_lab: np.ndarray
    z: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def total(self) -> np.ndarray:
        total: np.ndarray = self.x + self.y + self.z
        return total


class EOM:
    # pylint: disable=too-many-instance-attributes
    """Equations of motion for a disc.

    ``EOM`` is short for "equations of motion". Used to run the ODE solver
    from ``scipy``. It takes in a model for the disc, the trajectory object,
    the environment, and implements the functions for calculating forces
    and torques.

    Args:
        area: disc area
        I_xx: pitch and roll moments of inertia
        I_zz: spin moment of inertia
        mass: of the disc
        air_density: thiccness of the air
        g: gravitational acceleration
        model: must yield force and torque coefficients
    """

    def __init__(
        self,
        area: float,
        I_xx: float,
        I_zz: float,
        mass: float,
        air_density: float = 1.225,
        g: float = 9.81,
        # TODO: State handling of the model.
        # Also see constants.py
        # SEE: https://github.com/crzdg/flying-discs/pull/16#discussion_r2991607292
        model: Model = Model(),
    ):
        """Constructor."""
        self.area = area
        self.diameter = 2 * np.sqrt(self.area / np.pi)
        self.I_xx = I_xx
        self.I_zz = I_zz
        self.mass = mass
        self.model = model
        self.air_density = air_density
        self.g = g
        # Pre-compute some values to optimize the ODEs
        self.force_per_v2 = 0.5 * self.air_density * self.area  # N / (m/s)^2
        self.torque_per_v2 = self.force_per_v2 * self.diameter  # N * m / (m/s)^2
        self.grav = np.array([0, 0, -self.mass * self.g])
        self.z_hat = np.array([0, 0, 1])

    @classmethod
    def rotation_matrix_from_phi_theta(cls, phi: float, theta: float) -> np.ndarray:
        """Rotation matrix."""
        sp, cp = np.sin(phi), np.cos(phi)
        st, ct = np.sin(theta), np.cos(theta)
        return cls.rotation_matrix(sp, cp, st, ct)

    @staticmethod
    def rotation_matrix(sp: float, cp: float, st: float, ct: float) -> np.ndarray:
        """Compute the rotation matrix.

        Compute the (partial) rotation matrix that transforms from the
        lab frame to the disc frame. Note that because of azimuthal
        symmetry, the azimuthal angle (`gamma`) is not used.

        This matrix (R) can be used to transform a vector from the lab frame (L)
        into the disk frame (D), i.e.: r_D = R dot r_L.

        The ``z_hat`` unit vector in the disk frame (D) will always be pointing
        perpendicular up from the top face of the disk.
        """
        return np.array([[ct, sp * st, -st * cp], [0, cp, sp], [st, -sp * ct, cp * ct]])

    @classmethod
    def compute_angle_of_attack(
        cls,
        phi: float,
        theta: float,
        velocity: np.ndarray,
    ) -> tuple[float, float, float, float, float, np.ndarray, float, np.ndarray]:
        """Compute the angle of attack."""
        # TODO: Does not guard against zhat (norm == 0)
        # SEE: https://github.com/crzdg/flying-discs/pull/16#discussion_r2991607301
        # Rotation matrix
        sp, cp = np.sin(phi), np.cos(phi)
        st, ct = np.sin(theta), np.cos(theta)
        rotation_matrix = cls.rotation_matrix(sp, cp, st, ct)
        # Unit vectors
        zhat = rotation_matrix[2]
        v_dot_zhat = velocity @ zhat
        v_in_plane = velocity - zhat * v_dot_zhat
        angle_of_attack = -np.arctan(v_dot_zhat / np.linalg.norm(v_in_plane))
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
        phi: float,
        theta: float,
        velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> _GeometricQuantitiesResult:
        # pylint: disable=too-many-locals
        """Compute various vectors and pseudo vectors from the euler angles.

        Compute intermediate quantities on the way to computing the time
        derivatives of the kinematic variables.
        """
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
        # TODO: Does not guard against v_is_in_plane ~0
        # SEE: https://github.com/crzdg/flying-discs/pull/16#discussion_r2991607316
        zhat: np.ndarray = rotation_matrix[2]
        xhat: np.ndarray = v_in_plane / np.linalg.norm(v_in_plane)
        yhat = np.cross(zhat, xhat)
        w_prime = np.array(
            [
                angular_velocity[0] * ct,
                angular_velocity[1],
                angular_velocity[0] * st + angular_velocity[2],
            ]
        )
        # Angular velocity in lab coordinates
        w_lab = w_prime @ rotation_matrix
        # Angular velocity components along the unit vectors in the lab frame
        w = np.array([xhat, yhat, zhat]) @ w_lab
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
        velocity: np.ndarray,
        geometric_quantities: _GeometricQuantitiesResult,
    ) -> _ForcesResult:
        """Computes the lift, drag, and gravitational forces on the disc."""
        # TODO: Does not handle || velocity || = 0.
        # SEE: https://github.com/crzdg/flying-discs/pull/16#discussion_r2991607198
        vhat = velocity / np.linalg.norm(velocity)
        force_amplitude = self.force_per_v2 * (velocity @ velocity)
        # Compute the lift and drag forces
        res = _ForcesResult(
            lift=self.model.lift(geometric_quantities.angle_of_attack)
            * force_amplitude
            * np.cross(vhat, geometric_quantities.yhat),
            drag=self.model.drag(geometric_quantities.angle_of_attack) * force_amplitude * (-vhat),
            grav=self.grav,
            mass=self.mass,
        )
        return res

    def compute_torques(
        self,
        velocity: np.ndarray,
        geometric_quantities: _GeometricQuantitiesResult,
    ) -> _TorquesResult:
        """Computes the torque around each principle axis."""
        torque_amplitude = self.torque_per_v2 * (velocity @ velocity)
        wx, wy, wz = geometric_quantities.w
        # Compute component torques. Note that "x" and "y" are computed
        # in the lab frame
        t_x_lab = self.model.x(wx, wz) * torque_amplitude * geometric_quantities.xhat
        t_y_lab = self.model.y(geometric_quantities.angle_of_attack, wy) * torque_amplitude * geometric_quantities.yhat
        z = self.model.z(wz) * torque_amplitude * self.z_hat
        x = geometric_quantities.rotation_matrix @ t_x_lab
        y = geometric_quantities.rotation_matrix @ t_y_lab
        return _TorquesResult(x_lab=t_x_lab, y_lab=t_y_lab, z=z, x=x, y=y)

    def compute_derivatives(self, _: float, coordinates: np.ndarray) -> np.ndarray:
        """Right hand side of the ordinary differential equations.

        This is supplied to :meth:`scipy.integrate.solve_ivp`. See `this page
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp>`_
        for more information about its `fun` argument.

        .. todo::

           Implement the disc hitting the ground as a (Callable) scipy
           event object.

        Args:
          coordinates (np.ndarray): kinematic variables of the disc

        Returns:
          derivatives of all coordinates
        """
        # If the disk hit the ground, then stop calculations
        if coordinates[2] <= 0:
            return coordinates * 0

        velocity = coordinates[3:6]
        ang_velocity = coordinates[9:12]
        geometric_quantities = self.geometric_quantities(coordinates[6], coordinates[7], velocity, ang_velocity)
        forces = self.compute_forces(velocity, geometric_quantities)
        torques = self.compute_torques(velocity, geometric_quantities)
        return np.concatenate((velocity, forces.acc, ang_velocity, torques.total))
