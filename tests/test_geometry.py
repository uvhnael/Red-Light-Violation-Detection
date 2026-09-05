"""Test thuần hình học cho tripwire (side_of_line + segments_intersect).

Đã từng có bug liên quan:
* signed_distance_to_line raise khi 2 đầu mút trùng nhau (đã được assert).
* segments_intersect bỏ sót giao điểm tại biên (collinear overlap).
* side_of_line với deadband trả 0 cho mọi điểm gần line.
"""
from __future__ import annotations

import pytest

from edge_node.core.contracts import Point
from edge_node.core.geometry import (
    cross,
    segments_intersect,
    side_of_line,
    signed_distance_to_line,
)


class TestSideOfLine:
    """Vạch ngang y=400, hướng start->end từ trái qua phải.

    Image-space coords: y TRỎ XUỐNG. Vạch ngang y=400, điểm có y < 400
    nằm phía trên vạch trong ảnh. Công thức cross-product trả về giá trị
    mà *không trực tiếp* map sang "trên/dưới" mà chỉ phân hai bên — ký hiệu
    ±1 phụ thuộc vào quy ước hệ toán. Test dưới đây chỉ assert "hai điểm
    khác bên cho ra ký hiệu khác nhau" và "deadband giữa cho ra 0".
    """

    def setup_method(self) -> None:
        self.start = Point(0.0, 400.0)
        self.end = Point(1000.0, 400.0)

    def test_two_sides_have_opposite_signs(self):
        above = side_of_line(Point(500, 100), self.start, self.end)
        below = side_of_line(Point(500, 800), self.start, self.end)
        assert above == -below, f"above={above} below={below}, không đối xứng"
        assert above != 0

    def test_on_line_is_zero(self):
        assert side_of_line(Point(500, 400), self.start, self.end) == 0

    def test_deadband_inside_is_zero(self):
        # Vạch y=400 ± 3px → 397..403 đều = 0
        assert side_of_line(Point(500, 403), self.start, self.end, deadband_px=3.0) == 0
        assert side_of_line(Point(500, 397), self.start, self.end, deadband_px=3.0) == 0

    def test_deadband_outside_is_nonzero(self):
        above_band = side_of_line(Point(500, 404), self.start, self.end, deadband_px=3.0)
        below_band = side_of_line(Point(500, 396), self.start, self.end, deadband_px=3.0)
        assert above_band != 0
        assert below_band != 0
        assert above_band == -below_band


class TestSignedDistance:
    def test_horizontal_line(self):
        start = Point(0.0, 100.0)
        end = Point(100.0, 100.0)
        # Điểm (50, 50) cách đường y=100 đúng 50 pixel.
        # cross = (100-0)*(50-100) - (100-100)*(50-0) = 100*(-50) - 0 = -5000
        # distance = -5000 / length(100) = -50
        d = signed_distance_to_line(Point(50.0, 50.0), start, end)
        assert d == pytest.approx(-50.0)

    def test_vertical_line(self):
        start = Point(50.0, 0.0)
        end = Point(50.0, 100.0)
        # Điểm (100, 50) bên phải đường đứng.
        # cross = (b-a)×(c-a) = (0,100)×(50,50) = 0*50 - 100*50 = -5000
        # distance = -5000 / 100 = -50
        d = signed_distance_to_line(Point(100.0, 50.0), start, end)
        assert d == pytest.approx(-50.0)

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="identical"):
            signed_distance_to_line(Point(50, 50), Point(50, 50), Point(50, 50))


class TestSegmentsIntersect:
    """Cross-product based, kèm collinear-on-segment check."""

    def test_simple_cross(self):
        # Vạch ngang (a→b) giao vạch dọc (c→d)
        a, b = Point(0, 50), Point(100, 50)
        c, d = Point(50, 0), Point(50, 100)
        assert segments_intersect(a, b, c, d)

    def test_parallel_no_intersect(self):
        a, b = Point(0, 0), Point(100, 0)
        c, d = Point(0, 50), Point(100, 50)
        assert not segments_intersect(a, b, c, d)

    def test_touching_at_endpoint(self):
        # Hai đoạn chạm nhau tại đầu mút — vẫn coi là intersect.
        a, b = Point(0, 0), Point(10, 10)
        c, d = Point(10, 10), Point(20, 0)
        assert segments_intersect(a, b, c, d)

    def test_collinear_overlap(self):
        # Hai đoạn cùng nằm trên đường y=0, có overlap.
        a, b = Point(0, 0), Point(100, 0)
        c, d = Point(50, 0), Point(150, 0)
        assert segments_intersect(a, b, c, d)

    def test_collinear_no_overlap(self):
        a, b = Point(0, 0), Point(50, 0)
        c, d = Point(60, 0), Point(100, 0)
        assert not segments_intersect(a, b, c, d)

    def test_no_intersect_segments_apart(self):
        a, b = Point(0, 0), Point(10, 10)
        c, d = Point(20, 0), Point(30, 10)
        assert not segments_intersect(a, b, c, d)


class TestCross:
    """Hàm nhỏ, nhưng dùng để xác định hướng — đáng test."""

    def test_positive_cross(self):
        # a→b đi từ trái qua phải, c nằm trên → cross > 0
        a, b, c = Point(0, 0), Point(10, 0), Point(5, 5)
        assert cross(a, b, c) > 0

    def test_negative_cross(self):
        a, b, c = Point(0, 0), Point(10, 0), Point(5, -5)
        assert cross(a, b, c) < 0