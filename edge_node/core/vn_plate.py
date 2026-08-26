"""Quy tắc định dạng biển số xe Việt Nam (validator + chuẩn hoá).

Định dạng biển số Việt Nam (Thông tư 24/2023/TT-BCA):

    [Tỉnh 11-99]-[Serie: 2 chữ | 1 chữ + 1 số | 1 chữ][Số thứ tự: 4-5 số]

Tổng cộng 8-9 ký tự. Dạng chuẩn hoá (canonical): gạch ngang ngay sau
2 số tỉnh. Ví dụ:
    29-H12345    (Hà Nội, ô tô)
    30-K12345    (ô tô 5 số)
    59-X123456   (TP.HCM, mô tô 1 chữ + 1 số + 5 số)
    43-AB6789    (Đà Nẵng, mô tô)
    29-H1234     (mô tô 4 số)

Module dùng để:
* kiểm tra đầu ra OCR trước khi gắn vào violation event;
* chuẩn hoá về dạng ``NN-XXNNNNN`` (bỏ dấu chấm/khoảng trắng, giữ đúng
  một gạch sau mã tỉnh);
* "sửa" các lỗi OCR phổ biến (O↔0, I↔1, S↔5...) khi chuỗi khớp cấu trúc
  biển Việt Nam.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Cấu trúc biển số
# ---------------------------------------------------------------------------

# Tỉnh/thành: hai chữ số trong khoảng 11-99 (chữ số đầu 1-9, thứ hai 0-9 —
# bao gồm các mã có số 0 như 30, 43, 51, 60...).
_PROVINCE = r"(?:[1-9][0-9])"

# Chữ cái trong seri biển VN: A-Z đầy đủ theo Thông tư.
_LETTER = r"[A-Z]"

# Phần seri: 2 chữ (VD: AB), 1 chữ + 1 số (VD: X1 — mô tô), hoặc 1 chữ
# (VD: H — ô tô 5 số).
_SERIES = rf"(?:{_LETTER}{{2}}|{_LETTER}[0-9]|{_LETTER})"

# Số thứ tự: 4 hoặc 5 chữ số.
_SERIAL = r"[0-9]{4,5}"

# Pattern cho chuỗi đã bỏ hết ký tự trang trí (chỉ còn chữ + số).
# Lookahead chốt số lượng chữ số còn lại sau phần seri:
#   '30H12345' -> seri 'H'  + 5 số (ô tô);
#   '29AB1234' -> seri 'AB' + 4 số;
#   '59X123456' -> regex thử nhánh '2 chữ' rồi '1 chữ + 1 số' trước nên
#   tách thành X1|23456 (mô tô) — đúng TT24;
#   '29H1234'  -> chỉ 'H|1234' hợp lệ ('H1|234' chỉ có 3 số).
_PLATE_CORE_RE = re.compile(
    rf"^{_PROVINCE}{_SERIES}(?=[0-9]{{4,5}}$)[0-9]+$"
)

# Bản đồ thay thế cho các lỗi OCR kinh điển khi "repair" chuỗi gần đúng.
_OCR_CONFUSION_MAP = {
    "O": "0", "Q": "0", "D": "0",   # chỉ áp dụng ở vị trí số (xem repair)
    "I": "1", "L": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "P": "9",
}


@dataclass(frozen=True)
class PlateValidation:
    """Kết quả kiểm tra/định dạng một biển số."""

    valid: bool
    formatted: Optional[str] = None  # dạng canonical '29-H12345'
    reason: str = ""


def strip_separators(text: str) -> str:
    """Bỏ mọi ký tự không phải A-Z0-9 ('29H-123.45' -> '29H12345')."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_vn_plate(text: str) -> bool:
    """True khi *text* là biển số VN hợp lệ (có hoặc không có dấu phân cách)."""
    return _PLATE_CORE_RE.match(strip_separators(text)) is not None


def format_plate(text: str) -> Optional[str]:
    """Chuẩn hoá thành '29-H12345' (gạch sau mã tỉnh). None nếu không hợp lệ."""
    core = strip_separators(text)
    m = _PLATE_CORE_RE.match(core)
    if not m:
        return None
    province = core[:2]
    rest = core[2:]
    return f"{province}-{rest}"


def repair_plate_text(raw: str) -> Optional[str]:
    """Thử sửa lỗi OCR phổ biến rồi format lại.

    Chiến lược: sinh các ứng cử viên bằng cách thay từng ký tự dễ nhầm bằng
    "hàng xóm" của nó, giữ nguyên mọi vị trí khác, rồi chọn phương án đầu
    tiên tạo ra biển hợp lệ.

    Quy tắc thay thế theo vùng:
    * mã tỉnh (idx 0-1): chỉ digit ↔ digit (O/Q/D→0, I/L→1, Z→2...);
    * seri (idx 2-3): digit→chữ (OCR đọc 'O' thành '0') hoặc chữ→digit
      (vị trí thứ hai của seri kiểu 'chữ+số');
    * số thứ tự (từ idx 4): chỉ chữ → số (O→0, I→1, S→5, B→8...).
    """
    core = strip_separators(raw)
    if not core or len(core) < 8 or len(core) > 9:
        return None

    # Bước 1: nếu đã hợp lệ thì trả luôn.
    if _PLATE_CORE_RE.match(core):
        return format_plate(core)

    # Bản đồ digit -> letter cho vùng seri (OCR hay đọc nhầm chiều ngược lại).
    _DIGIT_TO_LETTER = {"0": ("O",), "1": ("I", "L"), "5": ("S",), "8": ("B",)}

    def try_replace(idx: int, new_ch: str) -> Optional[str]:
        replaced = core[:idx] + new_ch + core[idx + 1 :]
        return replaced if _PLATE_CORE_RE.match(replaced) else None

    candidates: list[str] = []
    for idx, ch in enumerate(core):
        if idx < 2:
            # Vùng tỉnh: chỉ nhận digit — thử biến thể digit.
            if ch in _OCR_CONFUSION_MAP:
                cand = try_replace(idx, _OCR_CONFUSION_MAP[ch])
                if cand:
                    candidates.append(cand)
        elif idx < 4:
            # Vùng seri: digit→chữ (OCR đọc 'O' thành '0') hoặc chữ→digit
            # (vị trí thứ hai của seri 'chữ+số').
            if ch in _DIGIT_TO_LETTER:
                for repl_ch in _DIGIT_TO_LETTER[ch]:
                    cand = try_replace(idx, repl_ch)
                    if cand:
                        candidates.append(cand)
            elif ch in _OCR_CONFUSION_MAP:  # chữ dễ nhầm -> thử digit
                cand = try_replace(idx, _OCR_CONFUSION_MAP[ch])
                if cand:
                    candidates.append(cand)
        else:
            # Vùng số thứ tự: chỉ chữ -> số.
            if ch in _OCR_CONFUSION_MAP and not ch.isdigit():
                cand = try_replace(idx, _OCR_CONFUSION_MAP[ch])
                if cand:
                    candidates.append(cand)

    if candidates:
        return format_plate(candidates[0])
    return None


def validate_and_format(raw: str) -> PlateValidation:
    """Điểm vào chính: kiểm tra và chuẩn hoá đầu ra OCR.

    Thứ tự: format trực tiếp -> repair -> báo không hợp lệ kèm lý do.
    """
    core = strip_separators(raw)
    direct = format_plate(core)
    if direct:
        return PlateValidation(True, direct)

    repaired = repair_plate_text(core)
    if repaired:
        return PlateValidation(True, repaired, "repaired")

    if not core:
        return PlateValidation(False, None, "empty")
    if len(core) < 8 or len(core) > 9:
        return PlateValidation(False, None, f"length {len(core)} (expect 8-9)")
    return PlateValidation(False, None, f"structure mismatch: {core}")
