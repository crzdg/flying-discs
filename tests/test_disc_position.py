import pytest

from flying_discs.disc_position import DiscPosition


@pytest.fixture
def setup() -> DiscPosition:
    return DiscPosition(x=10.0, y=11.0, z=12.0, vx=1.0, vy=1.0, vz=1.1, ax=0.5, ay=0.5, az=0.5)


def test_ground(setup: DiscPosition) -> None:
    position = setup
    position.ground()

    assert position.x == 10.0
    assert position.y == 11.0
    assert position.z == 0.0
    assert position.vx == 0.0
    assert position.vy == 0.0
    assert position.vz == 0.0
    assert position.ax == 0.0
    assert position.ay == 0.0
    assert position.az == 0.0


def test_reset(setup: DiscPosition) -> None:
    position = setup
    position.reset()

    assert position.x == 0.0
    assert position.y == 0.0
    assert position.z == 0.0
    assert position.vx == 0.0
    assert position.vy == 0.0
    assert position.vz == 0.0
    assert position.ax == 0.0
    assert position.ay == 0.0
    assert position.az == 0.0


def test_position(setup: DiscPosition) -> None:
    position = setup
    position_kwargs = {
        "x": 10.0,
        "y": 11.0,
        "z": 12.0,
        "vx": 1.0,
        "vy": 1.0,
        "vz": 1.1,
        "ax": 0.5,
        "ay": 0.5,
        "az": 0.5,
    }
    new_position = position.position(**position_kwargs)
    assert id(new_position) != id(position)
    assert new_position == position
