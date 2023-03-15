import math
from abc import ABC


class MorrisonConstants(ABC):
    # pylint: disable = invalid-name, too-many-instance-attributes
    GRAVITY = -9.81
    MASS = 0.175  # in kg
    RHO = 1.23
    A = 0.0568
    CL0 = 0.15
    CLalpha = 1.4
    CD0 = 0.08
    CDalpha = 2.72
    ALPHA0 = -4

    @classmethod
    def CL(cls, alpha: float) -> float:
        return cls.CL0 + cls.CLalpha * math.radians(alpha)

    @classmethod
    def CD(cls, alpha: float) -> float:
        return cls.CD0 + cls.CDalpha * math.pow(math.radians(alpha - cls.ALPHA0), 2)


class MorrisonUltrastar(MorrisonConstants):
    # pylint: disable = invalid-name
    RADIUS = 0.14  # in m
    MASS = 0.175  # in kg
    # A = RADIUS**2 * math.pi


class MorrisonUltrastarCode(MorrisonUltrastar):
    # pylint: disable = invalid-name
    # The paper do use a different CL0 value in the code then in the showed calculations.
    CL0 = 0.1
