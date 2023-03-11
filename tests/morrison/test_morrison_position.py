import pytest
from pytest_mock.plugin import MockerFixture

from flying_discs.morrison.morrison_position import DiscMorrisonPosition


@pytest.fixture
def setup() -> DiscMorrisonPosition:
    return DiscMorrisonPosition(
        x=1.0,
        y=2.0,
        z=3.0,
        vx=4.0,
        vy=5.0,
        vz=6.0,
        ax=7.0,
        ay=8.0,
        az=9.0,
        d=10.0,
        vd=11.0,
        ad=12.0,
    )


def test_ground(setup: DiscMorrisonPosition) -> None:
    position = setup
    position.ground()
    assert position == DiscMorrisonPosition(x=1.0, y=2.0)


def test_reset(setup: DiscMorrisonPosition) -> None:
    position = setup
    position.reset()
    assert position == DiscMorrisonPosition()
