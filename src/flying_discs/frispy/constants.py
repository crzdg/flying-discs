from abc import ABC
from typing import Type

from flying_discs.frispy.lib.eom import EOM as frispy_eom
from flying_discs.frispy.lib.model import Model as frispy_model


class Constants(ABC):
    # pylint: disable = invalid-name
    # defaults to ultrastar and see-level
    AREA: float = 0.058556  # m ^ 2
    I_xx: float = 0.001219  # kg * m ^ 2
    I_zz: float = 0.002352  # kg * m ^ 2
    MASS: float = 0.175  # kg
    RH0: float = 1.225  # air density, kg / m ^ 3
    GRAVITY: float = 9.81  # m / s ^ 2
    # TODO: Handle this instaition and possible overwrites of the instance,
    # potentially breaking the desired immutability of the constants.
    # SEE: https://github.com/crzdg/flying-discs/pull/16#discussion_r2991607127
    MODEL: frispy_model = frispy_model()
    EOM: Type = frispy_eom


class FrispyUltrastarConstants(Constants):
    AREA: float = 0.058556  # m ^ 2
    I_xx: float = 0.001219  # kg * m ^ 2
    I_zz: float = 0.002352  # kg * m ^ 2
    MASS: float = 0.175  # kg
