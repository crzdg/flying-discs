import math
from unittest.mock import MagicMock

import pytest

from flying_discs.morrison.constants import MorrisonUltrastar
from flying_discs.morrison.linear import MorrisonLinearCalculator


@pytest.fixture
def setup() -> MorrisonLinearCalculator:
    return MorrisonLinearCalculator(MorrisonUltrastar())


def test_calculate_trajectory() -> None:
    # TODO: Add test
    pass


def test_calculate_trajectory_to_position(setup: MorrisonLinearCalculator) -> None:
    # TODO: Add test
    pass


def test_approximate_trajectory_to_target_x_y() -> None:
    # TODO: Add test
    pass
