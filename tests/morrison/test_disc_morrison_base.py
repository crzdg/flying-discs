from unittest.mock import MagicMock

import pytest

from flying_discs.morrison.disc_morrison_base import DiscMorrison
from flying_discs.morrison.morrison_constants import DiscMorrisonUltrastar
from flying_discs.morrison.morrison_position import DiscMorrisonPosition


@pytest.fixture
def setup() -> DiscMorrison:
    return DiscMorrison(DiscMorrisonUltrastar())


def test_calculate_trajectory_to_position(setup: DiscMorrison) -> None:
    disc = setup
    with pytest.raises(RuntimeError):
        disc.calculate_trajectory_to_position(10.0, 20.0, 0.33)


def test_calculate_trajectory(setup: DiscMorrison) -> None:
    disc = setup
    disc._calculate_trajectory = MagicMock(spec_set=disc._calculate_trajectory, return_value=[DiscMorrisonPosition()])
    disc.calculate_trajectory(0.033, alpha=10.0, power=20.0)

    assert disc.throw_aoa == 10.0
    assert disc.throw_power == 20.0
    disc._calculate_trajectory.assert_called_once_with(20.0, 10.0, 0.033)
    assert disc.trajectory == [DiscMorrisonPosition()]


def test__calculate_trajectory(setup: DiscMorrison) -> None:
    disc = setup
    disc._calculate_trajectory_step = MagicMock(
        spec_set=disc._calculate_trajectory_step,
        side_effect=[
            DiscMorrisonPosition(z=1.0),
            DiscMorrisonPosition(z=0.1),
            DiscMorrisonPosition(z=0.0),
            DiscMorrisonPosition(z=-1.0),
        ],
    )

    trajectory = disc._calculate_trajectory(20.0, 10.0, 0.033)

    assert trajectory[0] == DiscMorrisonPosition(z=1.7, vy=20.0, vd=20.0)
    assert trajectory[1] == DiscMorrisonPosition(z=1.0)
    assert trajectory[2] == DiscMorrisonPosition(z=0.1)
    assert trajectory[3] == DiscMorrisonPosition(z=0.0)
    assert len(trajectory) == 4

    assert len(disc._calculate_trajectory_step.mock_calls) == 3
