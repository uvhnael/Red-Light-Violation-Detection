# -*- coding: utf-8 -*-
"""Build báo cáo .docx từ spec JSON — ghép content_1 + content_2."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from report_content_1 import BLOCKS_1
from report_content_2 import BLOCKS_2

spec = {
    "page": {"width_mm": 210, "height_mm": 297,
             "margins_mm": {"top": 25, "bottom": 25, "left": 30, "right": 20}},
    "header": "Báo cáo đồ án — Hệ thống phát hiện vi phạm vượt đèn đỏ (RLVD)",
    "footer": "Đồ án RLVD — Tháng 8/2026",
    "footer_page_numbers": True,
    "styles": [
        {"name": "Body", "base": "Normal", "font": "Times New Roman", "size_pt": 13,
         "color": "1A1A1A"},
        {"name": "BodyJ", "base": "Body", "font": "Times New Roman", "size_pt": 13},
        {"name": "CaptionV", "base": "Normal", "font": "Times New Roman", "size_pt": 11,
         "italic": True, "color": "555555"},
        {"name": "TitleBig", "base": "Normal", "font": "Times New Roman", "size_pt": 22,
         "bold": True, "color": "1F3864"},
        {"name": "TitleSub", "base": "Normal", "font": "Times New Roman", "size_pt": 16,
         "bold": True, "color": "2E5395"},
        {"name": "TitleSub2", "base": "Normal", "font": "Times New Roman", "size_pt": 13,
         "color": "444444"},
        {"name": "TitleMeta", "base": "Normal", "font": "Times New Roman", "size_pt": 12,
         "color": "666666"},
    ],
    "blocks": BLOCKS_1 + BLOCKS_2,
}

out = HERE / "RLVD_BaoCao_DoAn.docx"
(HERE / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1),
                                encoding="utf-8")
print("spec.json written:", (HERE / "spec.json").stat().st_size, "bytes")
print("out target:", out)
