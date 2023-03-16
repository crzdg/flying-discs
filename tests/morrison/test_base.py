from flying_discs.morrison.base import MorrisonBaseCalculator, MorrisonBaseThrow
from flying_discs.morrison.constants import MorrisonUltrastar
from flying_discs.morrison.coordinates import MorrisonPosition2D, MorrisonTrajectory2D


def test_calculate_trajectory_step() -> None:
    constants = MorrisonUltrastar()
    disc = MorrisonBaseCalculator(constants)
    angle_of_attack = 2.5
    timescale = 0.1
    CD = disc.constants.CL(angle_of_attack)
    CL = disc.constants.CD(angle_of_attack)
    next_step = disc.calculate_trajectory_step(0, 1, 10, 0, CD, CL, timescale)

    assert next_step == MorrisonPosition2D(
        0.9852526510998534,
        0.9248566384843551,
        9.852526510998533,
        -0.7514336151564497,
        -0.14747348900146615,
        -0.7514336151564497,
    )


def test_calculate_trajectory() -> None:
    constants = MorrisonUltrastar()
    disc = MorrisonBaseCalculator(constants)
    angle_of_attack = 2.5
    timescale = 0.1
    z0 = 1
    v0 = 10
    throw = disc.calculate_trajectory(z0, v0, angle_of_attack, timescale)

    expected_trajectory = [
        MorrisonPosition2D(x=0.0, z=z0, vx=v0, vz=0.0, ax=0.0, az=0.0),
        MorrisonPosition2D(
            x=0.9919651765304759,
            z=0.9440352825718474,
            vx=9.919651765304758,
            vz=-0.5596471742815253,
            ax=-0.08034823469524263,
            az=-0.5596471742815253,
        ),
        MorrisonPosition2D(
            x=1.9760241276525479,
            z=0.8314314687868644,
            vx=9.840589511220719,
            vz=-1.1260381378498308,
            ax=-0.07906225408403983,
            az=-0.5663909635683054,
        ),
        MorrisonPosition2D(
            x=2.9523023805468505,
            z=0.6615302836690609,
            vx=9.762782528943024,
            vz=-1.6990118511780343,
            ax=-0.07780698227769445,
            az=-0.5729737133282037,
        ),
        MorrisonPosition2D(
            x=3.920922488706201,
            z=0.4336890465086961,
            vx=9.686201081593506,
            vz=-2.2784123716036477,
            ax=-0.07658144734951824,
            az=-0.5794005204256135,
        ),
        MorrisonPosition2D(
            x=4.882004125307409,
            z=0.14728018121274578,
            vx=9.610816366012084,
            vz=-2.864088652959503,
            ax=-0.07538471558142223,
            az=-0.5856762813558555,
        ),
        MorrisonPosition2D(
            x=5.835664172941295,
            z=-0.19830925424677542,
            vx=9.536600476338858,
            vz=-3.4558943545952117,
            ax=-0.07421588967322611,
            az=-0.5918057016357084,
        ),
    ]

    assert throw == MorrisonBaseThrow(
        MorrisonTrajectory2D(expected_trajectory),
        constants,
        z0,
        v0,
        angle_of_attack,
        timescale,
    )
