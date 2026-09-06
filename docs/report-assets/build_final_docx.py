# -*- coding: utf-8 -*-
"""Build báo cáo đồ án .docx BẢN PRE-PRODUCTION từ final_content_1 + final_content_2."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from final_content_1 import BLOCKS_1
from final_content_2 import BLOCKS_2

spec = {
    "page": {"width_mm": 210, "height_mm": 297,
             "margins_mm": {"top": 25, "bottom": 25, "left": 30, "right": 20}},
    "header": "Báo cáo đồ án — Hệ thống phát hiện vi phạm vượt đèn đỏ (RLVD)",
    "footer": "Đồ án RLVD — Tháng 9/2026",
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

# Pre-check every image path exists (docx_create dies on FileNotFoundError)
missing = [b["path"] for b in spec["blocks"]
           if b.get("type") == "image" and not Path(b["path"]).exists()]
if missing:
    print("MISSING IMAGES:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

out = HERE / "final-report.docx"
(HERE / "final-spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
print("final-spec.json written:", (HERE / "final-spec.json").stat().st_size, "bytes")
print("out target:", out)
print("blocks:", len(spec["blocks"]))
