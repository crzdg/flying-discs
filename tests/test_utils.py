import math

import pytest
from numpy.testing import assert_almost_equal

from flying_discs.utils import (
    angle_between_vectors,
    distance_v1_v2,
    quadratic_bezier,
    rotate_points_around_index,
    rotate_points_around_mid_point,
    rotate_points_around_origin,
)


def test_angle_between_vectors() -> None:
    assert angle_between_vectors((10, 20), (20, 10)) == -angle_between_vectors((20, 10), (10, 20))
    assert angle_between_vectors((1, 0), (0, 1)) == math.radians(90)
    assert angle_between_vectors((0, 1), (1, 0)) == math.radians(-90)
    assert angle_between_vectors((0, 1), (0, -1)) == math.radians(180)
    assert angle_between_vectors((1, 0), (-1, 0)) == math.radians(180)


def test_distance_v1_v2() -> None:
    assert distance_v1_v2(*(10, 10), *(10, 10)) == 0
    assert distance_v1_v2(*(10, 10), *(0, 0)) == pytest.approx(14.142136)
    assert distance_v1_v2(*(10, 10), *(-10, -10)) == pytest.approx(28.284271)


def test_quadratic_bezier() -> None:
    p0 = (0.0, 0.0)
    p1 = (0.0, 2.0)
    p2 = (2.0, 2.0)

    # dt has to be in between [0,1]
    result = quadratic_bezier(0, p0, p1, p2)

    assert result == p0

    result = quadratic_bezier(1, p0, p1, p2)

    assert result == p2

    result = quadratic_bezier(0.5, p0, p1, p2)

    assert result == (0.5, 1.5)


def test_round_points_around_mid_origin() -> None:
    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_origin(testee, 180)
    expected = [(-1.0, -1.0), (-2.0, -2.0), (-3.0, -3.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_origin(testee, 90)
    expected = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_origin(testee, 270)
    expected = [(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_origin(testee, 360)

    assert_almost_equal(result, testee)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_origin(testee, 45)
    expected = [(0.0, 1.4142), (0.0, 2.8284), (0.0, 4.2426)]

    assert_almost_equal(result, expected, decimal=4)


def test_rotate_points_around_index() -> None:
    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 180, 0)
    expected = [(1.0, 1.0), (0.0, 0.0), (-1.0, -1.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 180, -1)
    expected = [(5.0, 5.0), (4.0, 4.0), (3.0, 3.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 90, 0)
    expected = [(1.0, 1.0), (0.0, 2.0), (-1.0, 3.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 90, -1)
    expected = [(5.0, 1.0), (4.0, 2.0), (3.0, 3.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 270, 0)
    expected = [(1.0, 1.0), (2.0, 0.0), (3.0, -1.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 270, -1)
    expected = [(1.0, 5.0), (2.0, 4.0), (3.0, 3.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 360, 0)

    assert_almost_equal(result, testee)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 360, -1)

    assert_almost_equal(result, testee)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 45, 0)
    expected = [(1.0, 1.0), (1.0, 2.4142), (1.0, 3.8284)]

    assert_almost_equal(result, expected, decimal=4)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_index(testee, 45, -1)
    expected = [(3.0, 0.1716), (3.0, 1.5858), (3.0, 3.0)]

    assert_almost_equal(result, expected, decimal=4)


def test_round_points_around_mid_point() -> None:
    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_mid_point(testee, 180)
    expected = [(3.0, 3.0), (2.0, 2.0), (1.0, 1.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_mid_point(testee, 90)
    expected = [(3.0, 1.0), (2.0, 2.0), (1.0, 3.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_mid_point(testee, 270)
    expected = [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]

    assert_almost_equal(result, expected)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_mid_point(testee, 360)

    assert_almost_equal(result, testee)

    testee = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = rotate_points_around_mid_point(testee, 45)
    expected = [(2.0, 0.5857), (2.0, 2.0), (2.0, 3.4142)]

    assert_almost_equal(result, expected, decimal=4)
