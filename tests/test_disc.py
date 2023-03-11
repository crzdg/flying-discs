from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock.plugin import MockerFixture

from flying_discs.disc import Disc
from flying_discs.disc_position import DiscPosition
from flying_discs.disc_state import DiscState


@pytest.fixture
def setup() -> Disc:
    class DiscTestImpl(Disc):
        def calculate_trajectory(self, timescale: float, **kwargs: Any) -> None:
            assert True

        def calculate_trajectory_to_position(self, x: float, y: float, timescale: float, **kwargs: Any) -> None:
            assert True

    return DiscTestImpl(DiscPosition(x=1.0, y=2.0, z=3.0), radius=0.28)


def test_distance_to_target(mocker: MockerFixture, setup: Disc) -> None:
    disc = setup
    disc.trajectory = [DiscPosition(x=2.0, y=3.0, z=4.0)]
    distance_v1_v2 = mocker.patch("flying_discs.disc.distance_v1_v2")

    disc.distance_to_target()
    distance_v1_v2.assert_called_once_with(disc.current_position.x, 2.0, disc.current_position.y, 3.0)


def test_position_properties(setup: Disc) -> None:
    disc = setup
    disc.current_position = DiscPosition(x=1.0, y=2.0, z=3.0, vx=4.0, vy=5.0, vz=6.0, ax=7.0, ay=8.0, az=9.0)

    assert disc.x == 1.0
    assert disc.y == 2.0
    assert disc.z == 3.0
    assert disc.vx == 4.0
    assert disc.vy == 5.0
    assert disc.vz == 6.0
    assert disc.ax == 7.0
    assert disc.ay == 8.0
    assert disc.az == 9.0


def test_traget_x_and_target_y(setup: Disc) -> None:
    disc = setup

    # Has no trajectory
    assert disc.target_x == 0.0
    assert disc.target_y == 0.0

    # Has trajectory
    trajectory = [
        DiscPosition(x=2.0, y=3.0, z=4.0),
        DiscPosition(x=3.0, y=4.0, z=0.0, ax=1.0),
    ]
    disc.trajectory = trajectory

    assert disc.target_x == 3.0
    assert disc.target_y == 4.0


def test_set_position(setup: Disc) -> None:
    disc = setup
    new_position = {"x": 10.0, "y": 12.0, "z": 14.0}
    disc.set_position(**new_position)
    assert disc.current_position == DiscPosition(x=10.0, y=12.0, z=14.0)


def test_move(mocker: MockerFixture, setup: Disc) -> None:
    disc = setup
    start_position = disc.current_position

    trajectory = [
        start_position,
        DiscPosition(x=2.0, y=3.0, z=4.0),
        DiscPosition(x=3.0, y=4.0, z=0.0, ax=1.0),
    ]
    disc.trajectory = trajectory
    disc.grounded = mocker.MagicMock(wraps=disc.grounded)

    # Does not move if not aired
    disc.move()
    assert disc.current_position == start_position

    # Does move when aired
    disc.release()
    disc.move()
    assert disc.current_step == 1
    assert disc.current_position == trajectory[1]

    # Does ground when z <= 0
    disc.move()
    disc.grounded.assert_called_once()
    assert disc.current_step == 0
    assert disc.state == DiscState.GROUNDED
    assert disc.trajectory == []
    assert disc.current_position == DiscPosition(x=3.0, y=4.0, z=0.0)


def test_release(setup: Disc) -> None:
    disc = setup
    disc.release()
    assert disc.state == DiscState.AIRED


def test_grounded(mocker: MockerFixture, setup: Disc) -> None:
    disc = setup
    position_mock = mocker.MagicMock(spec_set=DiscPosition)
    disc.current_position = position_mock
    disc._reset_disc_state = mocker.MagicMock()
    disc.grounded()

    position_mock.ground.assert_called_once()
    disc._reset_disc_state.assert_called_once()


def test_reset(mocker: MockerFixture, setup: Disc) -> None:
    disc = setup
    position_mock = mocker.MagicMock(spec_set=DiscPosition)
    disc.current_position = position_mock
    disc._reset_disc_state = mocker.MagicMock()
    disc.reset()

    position_mock.reset.assert_called_once()
    disc._reset_disc_state.assert_called_once()


def test__reset_disc_state(setup: Disc) -> None:
    disc = setup
    disc.release()
    disc.current_step = 10
    disc.trajectory = [DiscPosition()]
    disc._reset_disc_state()

    assert disc.state == DiscState.GROUNDED
    assert disc.current_step == 0
    assert disc.trajectory == []
