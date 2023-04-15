from abc import ABC
from typing import Type

import frispy


class Constants(ABC):
    # pylint: disable = invalid-name
    # defaults to ultrastar and see-level
    AREA: float = 0.058556  # m ^ 2
    I_xx: float = 0.001219  # kg * m ^ 2
    I_zz: float = 0.002352  # kg * m ^ 2
    MASS: float = 0.175  # kg
    RH0: float = 1.225  # air density, kg / m ^ 3
    GRAVITY: float = 9.81  # m / s ^ 2
    MODEL: frispy.model.Model = frispy.model.Model()
    EOM: Type = frispy.equations_of_motion.EOM


class FrispyUltrastarConstants(Constants):
    AREA: float = 0.058556  # m ^ 2
    I_xx: float = 0.001219  # kg * m ^ 2
    I_zz: float = 0.002352  # kg * m ^ 2
    MASS: float = 0.175  # kg
