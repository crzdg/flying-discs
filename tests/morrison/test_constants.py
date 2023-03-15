import pytest

from flying_discs.morrison.constants import MorrisonUltrastar


def test_CL() -> None:
    # expected result were obtained by manual calculation
    assert 0.21108652381980153 == pytest.approx(MorrisonUltrastar.CL(2.5))
    assert 0.17443460952792061 == pytest.approx(MorrisonUltrastar.CL(1))
    assert 0.15 == pytest.approx(MorrisonUltrastar.CL(0))
    assert 0.05226156188831754 == pytest.approx(MorrisonUltrastar.CL(-4))


def test_CD() -> None:
    # expected result were obtained by manual calculation
    assert 0.11500663388188856 == pytest.approx(MorrisonUltrastar.CD(2.5))
    assert 0.10071398454549618 == pytest.approx(MorrisonUltrastar.CD(1))
    assert 0.09325695010911755 == pytest.approx(MorrisonUltrastar.CD(0))
    assert 0.08 == pytest.approx(MorrisonUltrastar.CD(-4))
