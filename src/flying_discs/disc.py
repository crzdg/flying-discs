from abc import ABC, abstractmethod
from typing import Any, Sequence

from flying_discs.disc_position import DiscPosition
from flying_discs.disc_state import DiscState
from flying_discs.utils import distance_v1_v2


class Disc(ABC):
    # pylint: disable = invalid-name,too-many-instance-attributes,too-many-locals
    # TODO: manage radius, disc properties
    def __init__(self, position: DiscPosition, radius: float = 0.14) -> None:
        self.current_position = position
        self.radius = radius
        self.current_step = 0
        self.trajectory: Sequence[DiscPosition] = []
        self._state = DiscState.GROUNDED

    @abstractmethod
    def calculate_trajectory_to_position(
        self, x: float, y: float, timescale: float, **kwargs: Any
    ) -> Sequence[DiscPosition]:
        ...

    @abstractmethod
    def calculate_trajectory(self, timescale: float, **kwargs: Any) -> Sequence[DiscPosition]:
        ...

    @property
    def distance_to_target(self) -> float:
        return distance_v1_v2(self.x, self.target_x, self.y, self.target_y)

    @property
    def x(self) -> float:
        return self.current_position.x

    @property
    def y(self) -> float:
        return self.current_position.y

    @property
    def z(self) -> float:
        return self.current_position.z

    @property
    def vx(self) -> float:
        return self.current_position.vx

    @property
    def vy(self) -> float:
        return self.current_position.vy

    @property
    def vz(self) -> float:
        return self.current_position.vz

    @property
    def ax(self) -> float:
        return self.current_position.ax

    @property
    def ay(self) -> float:
        return self.current_position.ay

    @property
    def az(self) -> float:
        return self.current_position.az

    @property
    def state(self) -> DiscState:
        return self._state

    @property
    def target_x(self) -> float:
        if len(self.trajectory) > 0:
            return self.trajectory[-1].x
        # TODO: return none instead
        return 0.0

    @property
    def target_y(self) -> float:
        if len(self.trajectory) > 0:
            return self.trajectory[-1].y
        # TODO: return none instead
        return 0.0

    def set_position(self, **kwargs: Any) -> None:
        self.current_position = self.current_position.position(**kwargs)

    def move(self) -> None:
        if self._state == DiscState.AIRED:
            self.current_step += 1
            self.current_position = self.trajectory[self.current_step]
            if self.z <= 0.0:
                self.grounded()

    def release(self) -> None:
        self._state = DiscState.AIRED

    def grounded(self) -> None:
        self.current_position.ground()
        self._reset_disc_state()

    def reset(self) -> None:
        self.current_position.reset()
        self._reset_disc_state()

    def _reset_disc_state(self) -> None:
        self._state = DiscState.GROUNDED
        self.current_step = 0
        self.trajectory = []
