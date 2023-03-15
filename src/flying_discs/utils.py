import math
from typing import List, Tuple


def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """

    Returns the angle in radians between two vectors 'v1' and 'v2'

    Parameters
    ----------
    v1 : (float, float)
        First vector as tuple
    v2 : (float, float)
        Second vector as tuple

    Returns
    -------
    angle : float
        Angle between v1 and v2 in radians

    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle = math.atan2(det, dot)
    return angle


def distance_v1_v2(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Returns the distance of a two-dimensional point.

    Parameters
    ----------
    x1 : float
        First coordinate of the first point
    y1 : float
        Second coordinate of the first point
    x2 : float
        First coordinate of the second point
    y2 : float
        Second coordinate of the second point

    Returns
    -------
    distance : float
        Distance between two-dimensional coordinate (x1, y1) and (x2, y2)
    """
    dist_x = x1 - x2
    dist_y = y1 - y2
    dist = math.sqrt(dist_x**2 + dist_y**2)
    return dist


def quadratic_bezier(
    dt: float, p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculates a quadractic Bezier-curve.

    Args:
        t: Current time step
        p0: Point 0 of the Bezier curve
        p1: Point 1 of the Bezier curve
        p2: Point 2 of the Bezier curve

    Returns:
        The quadractic Bezier curve
    """
    # Calculate the coordinates of the point on the curve at parameter value t
    x = (1 - dt) ** 2 * p0[0] + 2 * (1 - dt) * dt * p1[0] + dt**2 * p2[0]
    y = (1 - dt) ** 2 * p0[1] + 2 * (1 - dt) * dt * p1[1] + dt**2 * p2[1]
    return (x, y)


def rotate_points_around_origin(points: List, theta: float) -> List[Tuple[float, float]]:
    """
    Rotates a 1D-list around its first point by the given angle.

    Args:
        points: List of values to rotate
        theta: Rotation angle

    Returns:
        Rotated list of values
    """
    theta = math.radians(theta)  # convert theta to radians
    theta_cos, theta_sin = math.cos(theta), math.sin(theta)
    rotated_points = []
    for point in points:
        x, y = point
        rotated_points.append((x * theta_cos - y * theta_sin, x * theta_sin + y * theta_cos))
    return rotated_points


def rotate_points_around_index(points: List, theta: float, index: int) -> List[Tuple[float, float]]:
    """
    Rotates a 1D-list around a given index by the given angle.


    Args:
        points: List of valeus to rotate
        theta: Rotation angle

    Returns:
        Rotated list of values
    """
    # get point at index
    x_at_index = points[index][0]
    y_at_index = points[index][1]

    # translate points to the origin
    points = [(x - x_at_index, y - y_at_index) for x, y in points]

    # rotate the points around the origin
    rotated_points = rotate_points_around_origin(points, theta)

    # translate it back
    rotated_points = [(x + x_at_index, y + y_at_index) for x, y in rotated_points]

    return rotated_points


def rotate_points_around_mid_point(points: list, theta: float) -> List[Tuple[float, float]]:
    """
    Rotates a 1D-list around its mid-point by a given angle.

    Args:
        points: List of values to rotate
        theta: Rotation angle

    Returns:
        Rotated list of values
    """
    # find the midpoint of the points
    x_mid = sum(x for x, _ in points) / len(points)
    y_mid = sum(y for _, y in points) / len(points)

    # translate the points to the origin
    points = [(x - x_mid, y - y_mid) for x, y in points]

    # rotate the points around the origin
    rotated_points = rotate_points_around_origin(points, theta)

    # translate the points back to the midpoint
    rotated_points = [(x + x_mid, y + y_mid) for x, y in rotated_points]

    return rotated_points
