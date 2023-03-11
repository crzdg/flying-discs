from unittest.mock import MagicMock

import pytest

from flying_discs.morrison.disc_morrison_bezier import DiscMorrisonBezier
from flying_discs.morrison.morrison_constants import DiscMorrisonUltrastar


@pytest.fixture
def setup() -> DiscMorrisonBezier:
    return DiscMorrisonBezier(DiscMorrisonUltrastar())


def test_calculate_trajectory(setup: DiscMorrisonBezier) -> None:
    disc = setup

    disc._calculate_trajectory = MagicMock(spec_set=disc._calculate_trajectory)

    disc.calculate_trajectory(0.033, direction=10.0, alpha=2.5, power=10.0, rotation_angle=34, factor=0.5)

    assert disc.throw_direction == 10.0
    assert disc.throw_aoa == 2.5
    assert disc.throw_power == 10.0
    assert disc.rotation_angle == 34
    assert disc.factor == 0.5

    disc._calculate_trajectory.assert_called_once_with(10.0, 2.5, 0.033, angle=10.0, rotation_angle=34, factor=0.5)


def test__calculate_trajectory() -> None:
    # TODO: Add test
    pass


def test__determine_bezier_points() -> None:
    # TODO: Add test
    pass
