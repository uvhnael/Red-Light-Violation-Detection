"""Geometry helpers for calibrated tripwire crossing."""

from __future__ import annotations

from math import hypot

from edge_node.core.contracts import Point


def cross(a: Point, b: Point, c: Point) -> float:
    """2D cross product of AB and AC."""

    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def signed_distance_to_line(point: Point, start: Point, end: Point) -> float:
    """Signed perpendicular distance from point to the oriented line start->end."""

    length = hypot(end.x - start.x, end.y - start.y)
    if length == 0.0:
        raise ValueError("Tripwire endpoints must not be identical")
    return cross(start, end, point) / length


def side_of_line(point: Point, start: Point, end: Point, deadband_px: float = 0.0) -> int:
    """Return -1, 0, or +1 for a point relative to an oriented line."""

    distance = signed_distance_to_line(point, start, end)
    if distance > deadband_px:
        return 1
    if distance < -deadband_px:
        return -1
    return 0


def _within(value: float, end_a: float, end_b: float, tolerance: float) -> bool:
    return min(end_a, end_b) - tolerance <= value <= max(end_a, end_b) + tolerance


def _on_segment(a: Point, b: Point, c: Point, tolerance: float) -> bool:
    return (
        abs(cross(a, b, c)) <= tolerance
        and _within(c.x, a.x, b.x, tolerance)
        and _within(c.y, a.y, b.y, tolerance)
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point, tolerance: float = 1e-6) -> bool:
    """Return whether segment AB intersects segment CD."""

    o1 = cross(a, b, c)
    o2 = cross(a, b, d)
    o3 = cross(c, d, a)
    o4 = cross(c, d, b)

    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if abs(o1) <= tolerance and _on_segment(a, b, c, tolerance):
        return True
    if abs(o2) <= tolerance and _on_segment(a, b, d, tolerance):
        return True
    if abs(o3) <= tolerance and _on_segment(c, d, a, tolerance):
        return True
    if abs(o4) <= tolerance and _on_segment(c, d, b, tolerance):
        return True
    return False
