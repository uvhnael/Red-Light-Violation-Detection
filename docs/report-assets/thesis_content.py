# -*- coding: utf-8 -*-
"""Đồ án RLVD — nội dung hoàn chỉnh, ghép từ ba phần.

build_thesis.py import BLOCKS, STYLES, PAGE từ module này.

Cấu trúc tài liệu:
  bìa → lời cảm ơn → tóm tắt → abstract → mục lục (hyperlink nội bộ)
  → danh sách hình / bảng / từ viết tắt
  → Chương 1 Giới thiệu          (thesis_part1)
  → Chương 2 Khảo sát và phân tích (thesis_part1)
  → Chương 3 Phân tích hệ thống    (thesis_part1)
  → Chương 4 Thiết kế hệ thống     (thesis_part2)
  → Chương 5 Cài đặt / Prototype   (thesis_part2)
  → Chương 6 Kiểm thử              (thesis_part3)
  → Chương 7 Kết luận              (thesis_part3)
  → Tài liệu tham khảo             (thesis_part3)
"""

from thesis_part1 import BLOCKS_1
from thesis_part2 import BLOCKS_2
from thesis_part3 import BLOCKS_3

BLOCKS = BLOCKS_1 + BLOCKS_2 + BLOCKS_3

# Khổ A4, lề trái 30 mm để đóng gáy theo quy cách báo cáo đồ án.
PAGE = {
    "width_mm": 210,
    "height_mm": 297,
    "margins_mm": {"top": 25, "bottom": 25, "left": 30, "right": 20},
}

FONT = "Times New Roman"

STYLES = [
    # ---- thân bài: hai biến thể, cùng cỡ chữ, khác canh lề
    {"name": "Body", "base": "Normal", "font": FONT, "size_pt": 13,
     "color": "1A1A1A", "space_after_pt": 6, "line_spacing": 1.3},
    {"name": "BodyJ", "base": "Body", "font": FONT, "size_pt": 13,
     "justify": True},
    # ---- đoạn in đậm dùng làm tiểu mục không đánh số
    {"name": "BodyBold", "base": "Body", "font": FONT, "size_pt": 13,
     "bold": True, "space_before_pt": 6},
    # ---- chú thích hình và bảng
    {"name": "CaptionV", "base": "Normal", "font": FONT, "size_pt": 11,
     "italic": True, "color": "555555", "space_before_pt": 2,
     "space_after_pt": 10, "keep_with_next": False},
    # ---- đoạn mã / lệnh, font đơn cách để phân biệt với thân bài
    {"name": "CodeV", "base": "Normal", "font": "Consolas", "size_pt": 10,
     "color": "1F3864", "space_before_pt": 4, "space_after_pt": 8,
     "left_indent_mm": 6},
    # ---- trang bìa
    {"name": "TitleBig", "base": "Normal", "font": FONT, "size_pt": 22,
     "bold": True, "color": "1F3864", "space_before_pt": 60,
     "space_after_pt": 10},
    {"name": "TitleSub", "base": "Normal", "font": FONT, "size_pt": 17,
     "bold": True, "color": "2E5395", "space_after_pt": 8},
    {"name": "TitleSub2", "base": "Normal", "font": FONT, "size_pt": 13,
     "color": "444444", "space_after_pt": 6},
    {"name": "TitleMeta", "base": "Normal", "font": FONT, "size_pt": 12,
     "color": "666666", "space_before_pt": 24, "space_after_pt": 4},
    # ---- một dòng trong mục lục tự dựng (hyperlink + PAGEREF)
    {"name": "TocEntry", "base": "Normal", "font": FONT, "size_pt": 13,
     "color": "0B3C7A", "space_after_pt": 3, "line_spacing": 1.15},
]
