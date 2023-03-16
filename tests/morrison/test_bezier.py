import math

from flying_discs.morrison.bezier import MorrisonBezierCalculator, MorrisonBezierThrow
from flying_discs.morrison.constants import MorrisonUltrastar
from flying_discs.morrison.coordinates import MorrisonPosition3D, MorrisonTrajectory3D


def test_calculate_trajectory() -> None:
    constants = MorrisonUltrastar()
    disc = MorrisonBezierCalculator(constants)
    angle_of_attack = 2.5
    timescale = 0.1
    z0 = 1
    v0 = 10
    direction_angle = math.radians(45)
    intersect_angle = math.radians(90)
    factor = 0.5
    initial_position = MorrisonPosition3D(0, 0, z0, 0, 0, 0, 0, 0, 0)
    throw = disc.calculate_trajectory(
        initial_position, v0, angle_of_attack, direction_angle, intersect_angle, factor, timescale
    )
    expected_positions = [
        MorrisonPosition3D(
            x=0.0, y=0.0, z=1, vx=7.0710678118654755, vy=7.071067811865475, vz=0.0, ax=0.0, ay=0.0, az=0.0
        ),
        MorrisonPosition3D(
            x=0.8793947789901058,
            y=0.8899903321439245,
            z=0.9440352825718474,
            vx=7.014253030256101,
            vy=7.0142530302561,
            vz=-0.5596471742815253,
            ax=-0.0568147816093743,
            ay=-0.056814781609374296,
            az=-0.5596471742815253,
        ),
        MorrisonPosition3D(
            x=1.6821274936784476,
            y=1.6990803787245574,
            z=0.8314314687868644,
            vx=6.958347574257383,
            vy=6.958347574257382,
            vz=-1.1260381378498308,
            ax=-0.05590545599871838,
            ay=-0.05590545599871837,
            az=-0.5663909635683054,
        ),
        MorrisonPosition3D(
            x=2.4081981440650253,
            y=2.427270139741899,
            z=0.6615302836690609,
            vx=6.903329729465163,
            vy=6.903329729465162,
            vz=-1.6990118511780343,
            ax=-0.05501784479221927,
            ay=-0.055017844792219266,
            az=-0.5729737133282037,
        ),
        MorrisonPosition3D(
            x=3.057606730149839,
            y=3.0745596151959487,
            z=0.4336890465086961,
            vx=6.849178468731238,
            vy=6.849178468731237,
            vz=-2.2784123716036477,
            ax=-0.054151260733924914,
            ay=-0.0541512607339249,
            az=-0.5794005204256135,
        ),
        MorrisonPosition3D(
            x=3.630353251932889,
            y=3.640948805086707,
            z=0.14728018121274578,
            vx=6.795873425145795,
            vy=6.795873425145794,
            vz=-2.864088652959503,
            ax=-0.05330504358544286,
            ay=-0.05330504358544285,
            az=-0.5856762813558555,
        ),
        MorrisonPosition3D(
            x=4.126437709414175,
            y=4.126437709414174,
            z=-0.19830925424677542,
            vx=6.7433948662860645,
            vy=6.743394866286064,
            vz=-3.4558943545952117,
            ax=-0.05247855885973085,
            ay=-0.052478558859730844,
            az=-0.5918057016357084,
        ),
    ]
    assert throw == MorrisonBezierThrow(
        MorrisonTrajectory3D(expected_positions),
        constants,
        initial_position,
        v0,
        angle_of_attack,
        0.7853981633974483,
        intersect_angle,
        factor,
        timescale,
    )


def test_calculate_trajectory_to_position() -> None:
    constants = MorrisonUltrastar()
    disc = MorrisonBezierCalculator(constants)
    angle_of_attack = 2.5
    timescale = 0.1
    z0 = 1
    intersect_angle = math.radians(90)
    factor = 0.5
    initial_position = MorrisonPosition3D(0, 0, z0, 0, 0, 0, 0, 0, 0)
    throw = disc.calculate_trajectory_to_position(
        initial_position, angle_of_attack, intersect_angle, factor, 10, 10, timescale
    )
    expected_positions = [
        MorrisonPosition3D(
            x=0.0, y=0.0, z=1, vx=10.32375900532357, vy=10.323759005323568, vz=0.0, ax=0.0, ay=0.0, az=0.0
        ),
        MorrisonPosition3D(
            x=1.4088005072214158,
            y=1.4331146129177537,
            z=0.9917155683301496,
            vx=10.202652616845029,
            vy=10.202652616845027,
            vz=-0.08284431669850374,
            ax=-0.12110638847854166,
            ay=-0.12110638847854163,
            az=-0.08284431669850374,
        ),
        MorrisonPosition3D(
            x=2.7359735598650987,
            y=2.7801810247675314,
            z=0.973051840219604,
            vx=10.084370922475369,
            vy=10.084370922475367,
            vz=-0.1866372811054564,
            ax=-0.1182816943696609,
            ay=-0.11828169436966089,
            az=-0.10379296440695268,
        ),
        MorrisonPosition3D(
            x=3.9815191579310483,
            y=4.041199235549332,
            z=0.9419866730766934,
            vx=9.968815864376221,
            vy=9.96881586437622,
            vz=-0.31065167142910666,
            ax=-0.11555505809914747,
            ay=-0.11555505809914746,
            az=-0.12401439032365023,
        ),
        MorrisonPosition3D(
            x=5.145437301419265,
            y=5.216169245263156,
            z=0.8965673096203468,
            vx=9.855893884108484,
            vy=9.855893884108482,
            vz=-0.45419363456346573,
            ax=-0.11292198026773673,
            ay=-0.11292198026773671,
            az=-0.14354196313435905,
        ),
        MorrisonPosition3D(
            x=6.227727990329748,
            y=6.305091053909004,
            z=0.8349072306504495,
            vx=9.745515666925133,
            vy=9.745515666925131,
            vz=-0.6166007896989737,
            ax=-0.11037821718335122,
            ay=-0.1103782171833512,
            az=-0.16240715513550796,
        ),
        MorrisonPosition3D(
            x=7.228391224662498,
            y=7.307964661486876,
            z=0.7551831846463601,
            vx=9.637595903337948,
            vy=9.637595903337946,
            vz=-0.7972404600408931,
            ax=-0.1079197635871862,
            ay=-0.10791976358718619,
            az=-0.18063967034191944,
        ),
        MorrisonPosition3D(
            x=8.147427004417516,
            y=8.224790067996771,
            z=0.6556323823827488,
            vx=9.532053066609855,
            vy=9.532053066609853,
            vz=-0.9955080226361134,
            ax=-0.1055428367280935,
            ay=-0.10554283672809348,
            az=-0.19826756259522027,
        ),
        MorrisonPosition3D(
            x=8.984835329594798,
            y=9.055567273438689,
            z=0.5345498456635145,
            vx=9.428809204943695,
            vy=9.428809204943693,
            vz=-1.2108253671923426,
            ax=-0.10324386166615952,
            ay=-0.10324386166615951,
            az=-0.21531734455622928,
        ),
        MorrisonPosition3D(
            x=9.740616200194347,
            y=9.800296277812631,
            z=0.3902859001065083,
            vx=9.327789747245602,
            vy=9.3277897472456,
            vz=-1.4426394555700617,
            ax=-0.10101945769809413,
            ay=-0.1010194576980941,
            az=-0.23181408837771908,
        ),
        MorrisonPosition3D(
            x=10.414769616216166,
            y=10.458977081118597,
            z=0.2212438026718603,
            vx=9.228923321438105,
            vy=9.228923321438103,
            vz=-1.69042097434648,
            ax=-0.09886642580749727,
            ay=-0.09886642580749726,
            az=-0.24778151877641819,
        ),
        MorrisonPosition3D(
            x=11.007295577660248,
            y=11.031609683356583,
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
    assert throw == MorrisonBezierThrow(
        MorrisonTrajectory3D(expected_positions),
        constants,
        initial_position,
        14.599999999999964,
        angle_of_attack,
        0.7853981633974483,
        intersect_angle,
        factor,
        timescale,
        target_x=10,
        target_y=10,
    )
