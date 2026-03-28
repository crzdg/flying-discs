from unittest import TestCase

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from flying_discs.frispy.jax_backend.eom import EOM


class TestEOM(TestCase):
    def setUp(self):
        super().setUp()
        # Coordinates for the default
        self.phi = 0
        self.theta = 0
        self.vel = jnp.array([1, 0, 0], dtype=jnp.float32)
        self.ang_vel = jnp.array([0, 0, 62], dtype=jnp.float32)
        self.kwargs = {
            "area": 0.058556,  # m^2
            "I_zz": 0.002352,  # kg*m^2
            "I_xx": 0.001219,  # kg*m^2
            "mass": 0.175,  # kg
        }

    def test_smoke(self):
        eom = EOM(**self.kwargs)
        assert eom is not None

    def test_eom_has_properties(self):
        eom = EOM(**self.kwargs)
        assert hasattr(eom, "model")

    def test_compute_forces(self):
        eom = EOM(**self.kwargs)
        geometric_quantities = eom.geometric_quantities(self.phi, self.theta, self.vel, self.ang_vel)
        result = eom.compute_forces(self.vel, geometric_quantities)
        assert result.lift.shape == (3,)
        assert jnp.issubdtype(result.lift.dtype, jnp.floating)
        assert result.drag.shape == (3,)
        assert jnp.issubdtype(result.drag.dtype, jnp.floating)
        assert result.grav.shape == (3,)
        assert jnp.issubdtype(result.grav.dtype, jnp.floating)
        assert result.total.shape == (3,)
        assert jnp.issubdtype(result.total.dtype, jnp.floating)
        assert result.acc.shape == (3,)
        assert jnp.issubdtype(result.acc.dtype, jnp.floating)

    def test_F_drag_direction(self):
        eom = EOM(**self.kwargs)
        geometric_quantities = eom.geometric_quantities(self.phi, self.theta, self.vel, self.ang_vel)
        result = eom.compute_forces(self.vel, geometric_quantities)
        assert result.drag[0] < 0  # backwards
        npt.assert_allclose(result.drag[1], 0, atol=1e-6)
        npt.assert_allclose(result.drag[2], 0, atol=1e-6)

    def test_F_lift_cross_component(self):
        eom = EOM(**self.kwargs)
        geometric_quantities = eom.geometric_quantities(self.phi, self.theta, self.vel, self.ang_vel)
        result = eom.compute_forces(self.vel, geometric_quantities)
        npt.assert_allclose(result.lift[1], 0, atol=1e-6)

    def test_F_grav_direction(self):
        eom = EOM(**self.kwargs)
        geometric_quantities = eom.geometric_quantities(self.phi, self.theta, self.vel, self.ang_vel)
        result = eom.compute_forces(self.vel, geometric_quantities)
        npt.assert_allclose(result.grav[0], 0, atol=1e-6)
        npt.assert_allclose(result.grav[1], 0, atol=1e-6)
        assert result.grav[2] < 0  # downwards

    def test_F_total_Acc_relation(self):
        eom = EOM(**self.kwargs)
        geometric_quantities = eom.geometric_quantities(self.phi, self.theta, self.vel, self.ang_vel)
        result = eom.compute_forces(self.vel, geometric_quantities)
        npt.assert_allclose(result.total, result.acc * eom.mass, rtol=1e-5)

    def test_compute_torques_smoke(self):
        eom = EOM(**self.kwargs)
        geometric_quantities = eom.geometric_quantities(self.phi, self.theta, self.vel, self.ang_vel)
        result = eom.compute_torques(self.vel, geometric_quantities)
        assert result.x_lab.shape == (3,)
        assert jnp.issubdtype(result.x_lab.dtype, jnp.floating)
        assert result.y_lab.shape == (3,)
        assert jnp.issubdtype(result.y_lab.dtype, jnp.floating)
        assert result.x.shape == (3,)
        assert jnp.issubdtype(result.x.dtype, jnp.floating)
        assert result.y.shape == (3,)
        assert jnp.issubdtype(result.y.dtype, jnp.floating)
        assert result.z.shape == (3,)
        assert jnp.issubdtype(result.z.dtype, jnp.floating)
        assert result.total.shape == (3,)
        assert jnp.issubdtype(result.total.dtype, jnp.floating)

    def test_compute_derivatives_smoke(self):
        eom = EOM(**self.kwargs)
        coords = jnp.array([0, 0, 1, 10, 0, 0, 0, 0, 0, 0, 0, 62], dtype=jnp.float32)
        der = eom.compute_derivatives(0, coords, None)
        assert der.shape == (12,)
        assert jnp.issubdtype(der.dtype, jnp.floating)

    def test_rotation_matrix(self):
        def trig_functions(phi, theta):
            return np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)

        r = np.eye(3)  # identity -- no rotation
        assert np.all(EOM.rotation_matrix(*trig_functions(0, 0)) == r)
        # 90 degrees counter clockwise around the primary "x" axis
        r = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        npt.assert_allclose(EOM.rotation_matrix(*trig_functions(np.pi / 2, 0)), r, atol=1e-6)
        # 90 degrees CCW around the secondary "y" axis
        r = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        npt.assert_allclose(EOM.rotation_matrix(*trig_functions(0, np.pi / 2)), r, atol=1e-6)
        # 90 degrees CCW around the primary "x" axis then
        # 90 degrees CCW around the secondary "y" axis
        # This permutes the coordinates once
        r = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        npt.assert_allclose(
            EOM.rotation_matrix(*trig_functions(np.pi / 2, np.pi / 2)),
            r,
            atol=1e-6,
        )

    def test_geometric_quantities_case1(self):
        t = EOM(**self.kwargs)
        w = jnp.array([0, 0, 1], dtype=jnp.float32)
        res = t.geometric_quantities(0, 0, jnp.array([1, 0, 0], dtype=jnp.float32), w)
        npt.assert_array_equal(res.w, w)
        npt.assert_array_equal(res.w_prime, w)
        npt.assert_array_equal(res.w_lab, w)
        npt.assert_allclose(res.rotation_matrix, np.eye(3), atol=1e-6)
        npt.assert_allclose(res.angle_of_attack, 0, atol=1e-6)
        npt.assert_array_equal(res.xhat, jnp.array([1, 0, 0]))
        npt.assert_allclose(res.yhat, jnp.array([0, 1, 0]), atol=1e-6)
        npt.assert_array_equal(res.zhat, jnp.array([0, 0, 1]))

    def test_geometric_quantities_case2(self):
        t = EOM(**self.kwargs)
        v = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=jnp.float32)
        w = jnp.array([0, 0, 1], dtype=jnp.float32)
        res = t.geometric_quantities(0, 0, v, w)
        npt.assert_array_equal(res.w, w)
        npt.assert_array_equal(res.w_prime, w)
        npt.assert_array_equal(res.w_lab, w)
        npt.assert_allclose(res.rotation_matrix, np.eye(3), atol=1e-6)
        npt.assert_allclose(res.angle_of_attack, 0, atol=1e-6)
        npt.assert_allclose(
            res.xhat,
            np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
            atol=1e-6,
        )
        npt.assert_allclose(
            res.yhat,
            np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
            atol=1e-6,
        )
        npt.assert_allclose(
            res.zhat,
            np.array([0, 0, 1]),
            atol=1e-6,
        )

    def test_geometric_quantities_case3(self):
        t = EOM(**self.kwargs)
        theta = np.pi / 4  # 45 degrees
        v = jnp.array([1, 0, 0], dtype=jnp.float32)
        w = jnp.array([0, 0, 1], dtype=jnp.float32)
        res = t.geometric_quantities(0, theta, v, w)
        npt.assert_allclose(res.w, w, atol=1e-6)
        npt.assert_array_equal(res.w_prime, w)
        npt.assert_allclose(res.w_lab, np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2)]), atol=1e-6)
        npt.assert_allclose(
            res.rotation_matrix,
            np.array(
                [
                    [1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
                    [0, 1, 0],
                    [1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
                ]
            ),
            atol=1e-6,
        )
        npt.assert_allclose(res.angle_of_attack, -np.pi / 4, atol=1e-5)
        npt.assert_allclose(
            res.xhat,
            np.array([1 / np.sqrt(2), 0, -1 / np.sqrt(2)]),
            atol=1e-6,
        )
        npt.assert_allclose(
            res.yhat,
            np.array([0, 1, 0]),
            atol=1e-6,
        )
        npt.assert_allclose(
            res.zhat,
            np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
            atol=1e-6,
        )
