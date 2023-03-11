import math
from unittest.mock import MagicMock

import pytest

from flying_discs.morrison.disc_morrison_linear import DiscMorrisonLinear
from flying_discs.morrison.morrison_constants import DiscMorrisonUltrastar


@pytest.fixture
def setup() -> DiscMorrisonLinear:
    return DiscMorrisonLinear(DiscMorrisonUltrastar())


def test_calculate_trajectory(setup: DiscMorrisonLinear) -> None:
    disc = setup

    disc._calculate_trajectory = MagicMock(spec_set=disc._calculate_trajectory)

    disc.calculate_trajectory(0.033, direction=10.0, alpha=2.5, v0=20.0)

    assert disc.throw_direction == 10.0
    assert disc.throw_aoa == 2.5
    assert disc.throw_power == 20.0

    disc._calculate_trajectory.assert_called_once_with(20.0, 2.5, 0.033, angle=10.0)


def test_calculate_trajectory_to_position(setup: DiscMorrisonLinear) -> None:
    disc = setup

    disc._approximate_trajectory_to_target_x_y = MagicMock(
        spec_set=disc._approximate_trajectory_to_target_x_y, return_value=([], 10.0, 2.5)
    )

    disc.calculate_trajectory_to_position(1, 1, 0.033)

    assert disc.throw_direction == math.radians(45)
    assert disc.throw_aoa == 2.5
    assert disc.throw_power == 10.0
    assert disc.trajectory == []

    disc._approximate_trajectory_to_target_x_y.assert_called_once_with(1, 1, 0.033, angle=math.radians(45))


def test__calculate_trajectory() -> None:
    # TODO: Add test
    pass


def test__approximate_trajectory_to_target_x_y() -> None:
    # TODO: Add test
    pass
