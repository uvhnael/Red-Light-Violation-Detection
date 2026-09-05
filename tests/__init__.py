"""Tests cho edge_node — pure-Python unit tests không cần GPU/UI.

Các module test:
* test_geometry — side_of_line, segments_intersect (đã từng được sửa logic
  deadband).
* test_violation_logic — ViolationDetector (deadband anchor fix 2026-08-28,
  crossing_point = bbox.center, no-tripwire gate, hysteresis).
* test_vn_plate — validate_and_format với 3 series shapes + repair OCR.

Không test:
* detector YOLO (cần model + GPU)
* pipeline main loop (đã có integration ở run_pipeline.py + ad-hoc script)
* api/server (FastAPI — đủ chậm để test riêng)
"""
