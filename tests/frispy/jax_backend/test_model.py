"""
Tests of the ``Model`` object (JAX implementation).
"""

from unittest import TestCase

import numpy as np

from flying_discs.frispy.jax_backend.model import Model


class TestModel(TestCase):
    def setUp(self):
        super().setUp()
        self.model = Model()

    def test_smoke(self):
        assert self.model is not None

    def test_value_updates(self):
        # Model is an eqx.Module (immutable); test that constructor accepts field overrides
        m = Model(PL0=1)
        assert m.PL0 == 1
        m = Model(PL0=2)
        assert m.PL0 == 2

    def test_lift(self):
        alphas = np.linspace(-1, 1)
        model = Model(PL0=1, PLa=1)
        cl = model.lift(alphas)
        # Linear, strictly increasing
        for i in range(1, len(alphas)):
            assert cl[i] - cl[i - 1] > 0

    def test_drag(self):
        alphas = np.linspace(-1, 1, 21)
        model = Model(PD0=1, PDa=1, alpha_0=0)
        cd = model.drag(alphas)
        # Quadratic, down then up
        for i in range(1, 11):
            assert cd[i] - cd[i - 1] < 0
        for i in range(11, 21):
            assert cd[i] - cd[i - 1] > 0

    def test_x(self):
        wx = np.linspace(-1, 1)
        wz = np.linspace(-1, 1)
        model = Model(PTxwx=1, PTxwz=1)
        cx = model.x(wx, 0)
        # Linear, strictly increasing
        for i in range(1, len(cx)):
            assert cx[i] - cx[i - 1] > 0
        cx = model.x(0, wz)
        # Linear, strictly increasing
        for i in range(1, len(cx)):
            assert cx[i] - cx[i - 1] > 0

    def test_y(self):
        alphas = np.linspace(-1, 1)
        wy = np.linspace(-1, 1)
        model = Model(PTy0=1, PTywy=1, PTya=1)
        cy = model.y(alphas, 0)
        # Linear, strictly increasing
        for i in range(1, len(cy)):
            assert cy[i] - cy[i - 1] > 0
        cy = model.y(0, wy)
        # Linear, strictly increasing
        for i in range(1, len(cy)):
            assert cy[i] - cy[i - 1] > 0
        # Intercept
        assert model.y(0, 0) == model.PTy0

    def test_z(self):
        wz = np.linspace(-1, 1)
        model = Model(PTzwz=1)
        cz = model.z(wz)
        # Linear, strictly increasing
        for i in range(1, len(cz)):
            assert cz[i] - cz[i - 1] > 0
        # Intercept
        assert model.z(0) == 0
