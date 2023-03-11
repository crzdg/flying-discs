from flying_discs.morrison.disc_morrison_linear import DiscMorrisonLinear
from flying_discs.morrison.morrison_constants import DiscMorrisonUltrastar

x0 = 0
y0 = 0
z0 = 1

disc = DiscMorrisonLinear(DiscMorrisonUltrastar(), x0, y0, z0)

timescale = 0.033
angle_of_attack = 5
v0 = 14
direction = 0

trajectory = disc.calculate_trajectory(
    timescale,
    alpha=angle_of_attack,
    v0=v0,
    direction=direction,
)
