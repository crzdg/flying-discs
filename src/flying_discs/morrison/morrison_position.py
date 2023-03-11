from dataclasses import dataclass

from flying_discs.disc_position import DiscPosition


@dataclass
class DiscMorrisonPosition(DiscPosition):
    # pylint: disable = invalid-name
    d: float = 0.0
    vd: float = 0.0
    ad: float = 0.0

    def ground(self) -> None:
        super().ground()
        self.d = 0.0
        self.vd = 0.0
        self.ad = 0.0
