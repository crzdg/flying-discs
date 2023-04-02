import math
from abc import ABC


class Constants(ABC):
    # pylint: disable = invalid-name
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


class MorrisonUltrastar(Constants):
    RADIUS = 0.14  # in m
    MASS = 0.175  # in kg
    # A = RADIUS**2 * math.pi


class MorrisonUltrastarCode(MorrisonUltrastar):
    # The paper do use a different CL0 value in the code then in the showed calculations.
    CL0 = 0.1


class HummelConstants(Constants):
    CL0 = 0.33
    CLalpha = 1.91
    CD0 = 0.18
    CDalpha = 0.69


class PottsCrowtherConstants(Constants):
    CL0 = 0.2
    CLalpha = 2.96
    CD0 = 0.08
    CDalpha = 2.72


class YasudaConstants(Constants):
    CL0 = 0.08
    CLalpha = 2.4
    CD0 = 0.1
    CDalpha = 2.3


class StilleyCarstensConstants(Constants):
    CL0 = 0.15
    CLalpha = 2.8
    CD0 = 0.1
    CDalpha = 2.5


class HannahConstants(Constants):
    CL0 = 0.33
    CLalpha = 1.9
    CD0 = 0.18
    CDalpha = 0.69
