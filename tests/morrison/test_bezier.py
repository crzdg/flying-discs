from unittest.mock import MagicMock

import pytest

from flying_discs.morrison.bezier import MorrisonBezierCalculator
from flying_discs.morrison.constants import MorrisonUltrastar


@pytest.fixture
def setup() -> MorrisonBezierCalculator:
    return MorrisonBezierCalculator(MorrisonUltrastar())


def test_calculate_trajectory() -> None:
    # TODO: Add test
    pass


def test__determine_bezier_points() -> None:
    # TODO: Add test
    pass
