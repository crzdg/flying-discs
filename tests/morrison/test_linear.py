import math

from flying_discs.morrison.constants import MorrisonUltrastar
from flying_discs.morrison.coordinates import MorrisonPosition3D, MorrisonTrajectory3D
from flying_discs.morrison.linear import MorrisonLinearCalculator, MorrisonLinearThrow


def test_calculate_trajectory() -> None:
    constants = MorrisonUltrastar()
    disc = MorrisonLinearCalculator(constants)
    angle_of_attack = 2.5
    direction_angle = math.radians(90)
    timescale = 0.1
    z0 = 1
    v0 = 10
    initial_position = MorrisonPosition3D(0, 0, z0, 0, 0, 0, 0, 0, 0)
    throw = disc.calculate_trajectory(initial_position, v0, angle_of_attack, direction_angle, timescale)
    expected_positions = [
        MorrisonPosition3D(x=0, y=0, z=1, vx=6.123233995736766e-16, vy=10.0, vz=0.0, ax=0.0, ay=0.0, az=0.0),
        MorrisonPosition3D(
            x=6.074034891518432e-17,
            y=0.9919651765304759,
            z=0.9440352825718474,
            vx=6.074034891518432e-16,
            vy=9.919651765304758,
            vz=-0.5596471742815253,
            ax=-4.91991042183346e-18,
            ay=-0.08034823469524263,
            az=-0.5596471742815253,
        ),
        MorrisonPosition3D(
            x=1.2099658114838167e-16,
            y=1.9760241276525479,
            z=0.8314314687868644,
            vx=6.025623223319735e-16,
            vy=9.840589511220719,
            vz=-1.1260381378498308,
            ax=-4.841166819869707e-18,
            ay=-0.07906225408403983,
            az=-0.5663909635683054,
        ),
        MorrisonPosition3D(
            x=1.8077638302259056e-16,
            y=2.9523023805468505,
            z=0.6615302836690609,
            vx=5.977980187420888e-16,
            vy=9.762782528943024,
            vz=-1.6990118511780343,
            ax=-4.764303589884667e-18,
            ay=-0.07780698227769445,
            az=-0.5729737133282037,
        ),
        MorrisonPosition3D(
            x=2.4008725877494617e-16,
            y=3.920922488706201,
            z=0.4336890465086961,
            vx=5.931087575235559e-16,
            vy=9.686201081593506,
            vz=-2.2784123716036477,
            ax=-4.6892612185329534e-18,
            ay=-0.07658144734951824,
            az=-0.5794005204256135,
        ),
        MorrisonPosition3D(
            x=2.9893653627409464e-16,
            y=4.882004125307409,
            z=0.14728018121274578,
            vx=5.884927749914848e-16,
            vy=9.610816366012084,
            vz=-2.864088652959503,
            ax=-4.615982532071117e-18,
            ay=-0.07538471558142223,
            az=-0.5856762813558555,
        ),
        MorrisonPosition3D(
            x=3.5733137251457216e-16,
            y=5.835664172941295,
            z=-0.19830925424677542,
            vx=5.839483624047753e-16,
            vy=9.536600476338858,
            vz=-3.4558943545952117,
            ax=-4.5444125867094725e-18,
            ay=-0.07421588967322611,
            az=-0.5918057016357084,
        ),
    ]
    assert throw == MorrisonLinearThrow(
        MorrisonTrajectory3D(expected_positions),
        constants,
        initial_position,
        v0,
        angle_of_attack,
        direction_angle,
        timescale,
    )


def test_calculate_trajectory_to_position() -> None:
    constants = MorrisonUltrastar()
    disc = MorrisonLinearCalculator(constants)
    angle_of_attack = 2.5
    timescale = 0.1
    z0 = 1
    initial_position = MorrisonPosition3D(0, 0, z0, 0, 0, 0, 0, 0, 0)
    throw = disc.calculate_trajectory_to_position(initial_position, angle_of_attack, 10, 10, timescale)

    expected_positions = [
        MorrisonPosition3D(x=0, y=0, z=1, vx=10.32375900532357, vy=10.323759005323568, vz=0.0, ax=0.0, ay=0.0, az=0.0),
        MorrisonPosition3D(
            x=1.020265261684503,
            y=1.0202652616845027,
            z=0.9917155683301496,
            vx=10.202652616845029,
            vy=10.202652616845027,
            vz=-0.08284431669850374,
            ax=-0.12110638847854166,
            ay=-0.12110638847854163,
            az=-0.08284431669850374,
        ),
        MorrisonPosition3D(
            x=2.02870235393204,
            y=2.0287023539320392,
            z=0.973051840219604,
            vx=10.084370922475369,
            vy=10.084370922475367,
            vz=-0.1866372811054564,
            ax=-0.1182816943696609,
            ay=-0.11828169436966089,
            az=-0.10379296440695268,
        ),
        MorrisonPosition3D(
            x=3.0255839403696623,
            y=3.0255839403696614,
            z=0.9419866730766934,
            vx=9.968815864376221,
            vy=9.96881586437622,
            vz=-0.31065167142910666,
            ax=-0.11555505809914747,
            ay=-0.11555505809914746,
            az=-0.12401439032365023,
        ),
        MorrisonPosition3D(
            x=4.011173328780511,
            y=4.01117332878051,
            z=0.8965673096203468,
            vx=9.855893884108484,
            vy=9.855893884108482,
            vz=-0.45419363456346573,
            ax=-0.11292198026773673,
            ay=-0.11292198026773671,
            az=-0.14354196313435905,
        ),
        MorrisonPosition3D(
            x=4.985724895473024,
            y=4.985724895473023,
            z=0.8349072306504495,
            vx=9.745515666925133,
            vy=9.745515666925131,
            vz=-0.6166007896989737,
            ax=-0.11037821718335122,
            ay=-0.1103782171833512,
            az=-0.16240715513550796,
        ),
        MorrisonPosition3D(
            x=5.949484485806819,
            y=5.949484485806818,
            z=0.7551831846463601,
            vx=9.637595903337948,
            vy=9.637595903337946,
            vz=-0.7972404600408931,
            ax=-0.1079197635871862,
            ay=-0.10791976358718619,
            az=-0.18063967034191944,
        ),
        MorrisonPosition3D(
            x=6.902689792467804,
            y=6.902689792467803,
            z=0.6556323823827488,
            vx=9.532053066609855,
            vy=9.532053066609853,
            vz=-0.9955080226361134,
            ax=-0.1055428367280935,
            ay=-0.10554283672809348,
            az=-0.19826756259522027,
        ),
        MorrisonPosition3D(
            x=7.845570712962173,
            y=7.845570712962172,
            z=0.5345498456635145,
            vx=9.428809204943695,
            vy=9.428809204943693,
            vz=-1.2108253671923426,
            ax=-0.10324386166615952,
            ay=-0.10324386166615951,
            az=-0.21531734455622928,
        ),
        MorrisonPosition3D(
            x=8.778349687686733,
            y=8.778349687686733,
            z=0.3902859001065083,
            vx=9.327789747245602,
            vy=9.3277897472456,
            vz=-1.4426394555700617,
            ax=-0.10101945769809413,
            ay=-0.1010194576980941,
            az=-0.23181408837771908,
        ),
        MorrisonPosition3D(
            x=9.701242019830543,
            y=9.701242019830543,
            z=0.2212438026718603,
            vx=9.228923321438105,
            vy=9.228923321438103,
            vz=-1.69042097434648,
            ax=-0.09886642580749727,
            ay=-0.09886642580749726,
            az=-0.24778151877641819,
        ),
        MorrisonPosition3D(
            x=10.614456178269114,
            y=10.614456178269112,
            z=0.025877495321820376,
            vx=9.1321415843857,
            vy=9.132141584385698,
            vz=-1.9536630735003992,
            ax=-0.09678173705240442,
            ay=-0.0967817370524044,
            az=-0.2632420991539194,
        ),
        MorrisonPosition3D(
            x=11.518194084526598,
            y=11.518194084526597,
            z=-0.19731052316364717,
            vx=9.037379062574846,
            vy=9.037379062574844,
            vz=-2.2318801848546754,
            ax=-0.09476252181085419,
            ay=-0.09476252181085418,
            az=-0.27821711135427635,
        ),
    ]
    assert throw == MorrisonLinearThrow(
        MorrisonTrajectory3D(expected_positions),
        constants,
        initial_position,
        14.599999999999964,
        angle_of_attack,
        0.7853981633974483,
        timescale,
        10,
        10,
    )
