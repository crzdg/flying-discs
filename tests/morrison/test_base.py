from unittest.mock import MagicMock

import pytest

from flying_discs.morrison.base import MorrisonBaseCalculator, MorrisonPosition2D
from flying_discs.morrison.constants import MorrisonUltrastar


@pytest.fixture
def setup() -> MorrisonBaseCalculator:
    return MorrisonBaseCalculator(MorrisonUltrastar())


def test_calculate_trajectory_step(setup: MorrisonBaseCalculator) -> None:
    disc = setup
    angle_of_attack = 2.5
    timescale = 0.1
    CD = disc.constants.CL(angle_of_attack)
    CL = disc.constants.CD(angle_of_attack)
    next_step = disc.calculate_trajectory_step(0, 1, 10, 0, CD, CL, timescale)

    assert next_step == MorrisonPosition2D(
        0.9852526510998534,
        0.9248566384843551,
        9.852526510998533,
        -0.7514336151564497,
        -0.14747348900146615,
        -0.7514336151564497,
    )


def test_calculate_trajectory(setup: MorrisonBaseCalculator) -> None:
    # TODO: Add test
    pass
