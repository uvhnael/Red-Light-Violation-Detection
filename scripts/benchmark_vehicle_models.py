#!/usr/bin/env python3
"""Benchmark dau: yolo26m_vehicle.pt (tự train) vs yolo26m.pt (mặc định COCO)
trên CÙNG tập valid của dataset vehicle, cùng cấu hình inference như edge_node
(conf=0.35, iou=0.45, imgsz=640).

So sánh:
  1. Chất lượng: mAP50 / mAP50-95 / P / R trên không gian class của dataset.
     Model COCO được map sang 4 lớp dataset:
       car(2)->car, motorcycle(3)+bicycle(1)->bike, bus(5)+truck(7)->van/bus|truck,
       train(6)->van/bus. Lưu ý COCO tách car/truck/bus nên mapping chỉ xấp xỉ —
     đây chính là hạn chế cần chứng minh bằng số.
  2. Tốc độ: FPS inference GPU FP16 (giống edge_node).
"""

import csv
import time
from pathlib import Path

import torch
from ultralytics import YOLO

BASE = Path("/home/uvhnael/projects/Red-Light-Violation-Detection")
DATA_YAML = BASE / "train_model/vehical_detection/data.yaml"
VALID_DIR = BASE / "train_model/vehical_detection/vehicle.v1i.yolo26/valid/images"
CUSTOM = BASE / "edge_node/models/yolo26m_vehicle.pt"
COCO = BASE / "edge_node/models/yolo26m.pt"
OUT_CSV = BASE / "scripts/benchmark_results.csv"

# COCO id -> dataset label (dataset: car, bike, van/bus, truck)
COCO_TO_DATASET = {
    1: "bike",        # bicycle
    2: "car",         # car
    3: "bike",        # motorcycle
    5: "van/bus",     # bus
    6: "van/bus",     # train (xấp xỉ)
    7: "truck",       # truck
}

# ── Ground truth từ label files ─────────────────────────────────────────────
def load_gt():
    """Trả về {img_stem: {label: count}} và tổng số box theo lớp."""
    gt_dir = VALID_DIR.parent / "labels"
    per_class_total = {0: 0, 1: 0, 2: 0, 3: 0}  # car bike van/bus truck
    n_img = 0
    for lf in sorted(gt_dir.glob("*.txt")):
        n_img += 1
        for line in lf.read_text().splitlines():
            parts = line.split()
            if parts:
                cid = int(float(parts[0]))
                if cid in per_class_total:
                    per_class_total[cid] += 1
    return n_img, per_class_total


def match_detections(model, conf=0.35, iou=0.45, max_imgs=None):
    """Chạy predict trên valid, trả về dict {cls_name: [conf,...]} cho det đúng ảnh-level.

    Đếm ở mức ảnh-level: mỗi det được gán nhãn lớp dự đoán; để so công bằng
    giữa 2 hệ class khác nhau, ta so theo NHÃN SAU MAP + đếm số det mỗi lớp
    trên toàn tập (phân bố dự đoán), kèm conf trung bình.
    """
    names_map = model.names
    stats = {}  # label -> [count, conf_sum]
    n_images = 0
    files = sorted(VALID_DIR.glob("*.*"))
    if max_imgs:
        files = files[:max_imgs]
    t0 = time.perf_counter()
    with torch.inference_mode():
        for f in files:
            r = model.predict(source=str(f), conf=conf, iou=iou, imgsz=640,
                              device="cuda:0", verbose=False, half=True)[0]
            n_images += 1
            if r.boxes is None:
                continue
            for cls_id, c in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                raw = names_map[int(cls_id)]
                if isinstance(raw, str) and not raw.isdigit():
                    # model custom: tên lớp trực tiếp; model coco: map
                    label = raw
                else:
                    continue
                if label in ("car", "bike", "van/bus", "truck"):
                    s = stats.setdefault(label, [0, 0.0])
                    s[0] += 1
                    s[1] += float(c)
    dt = time.perf_counter() - t0
    return stats, n_images, dt


def main():
    print("=" * 70)
    n_valid, gt_per_class = load_gt()
    labels = ["car", "bike", "van/bus", "truck"]
    print(f"Valid: {n_valid} anh | GT boxes theo lop: " +
          ", ".join(f"{labels[k]}={v}" for k, v in gt_per_class.items()))
    print("=" * 70)

    rows = []
    for tag, path in [("CUSTOM (yolo26m_vehicle)", CUSTOM), ("COCO (yolo26m mac dinh)", COCO)]:
        print(f"\n>>> Dang chay: {tag}")
        m = YOLO(str(path))
        # 1) chat luong tren valid (khong gian lop cua dataset)
        metrics = m.val(
            data=str(DATA_YAML), imgsz=640, batch=16,
            device="cuda:0", verbose=False, plots=False,
        )
        box = metrics.box
        mp, mr = float(box.mp), float(box.mr)
        map50, map5095 = float(box.map50), float(box.map)

        # per-class cho custom (class index khop voi dataset); coco thi bo qua
        per_class = {}
        try:
            for i, ci in enumerate(box.ap_class_index):
                per_class[m.names[int(ci)]] = (float(box.ap50[i].mean()), float(box.ap[i].mean()))
        except Exception:
            pass

        # 2) toc do FP16 nhu edge_node
        stats, n_used, dt = match_detections(m)
        fps = n_used / dt if dt > 0 else 0

        rows.append({
            "model": tag, "P": round(mp, 4), "R": round(mr, 4),
            "mAP50": round(map50, 4), "mAP50-95": round(map5095, 4),
            "FPS_FP16": round(fps, 1),
            **{f"det_{l}": stats.get(l, [0, 0])[0] for l in labels},
            **{f"conf_{l}": round(stats.get(l, [0, 0.0])[1] / max(stats.get(l, [1, 1])[0], 1), 3) for l in labels},
            "per_class": per_class,
        })
        del m
        torch.cuda.empty_cache()

    # ── In bang ket qua ──
    print("\n" + "=" * 70)
    print("KET QUA SO SANH (valid 1980 anh, conf=0.35 iou=0.45 imgsz=640)")
    print("=" * 70)
    hdr = f"{'Model':<28}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}{'FPS':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['model']:<28}{r['P']:>7.4f}{r['R']:>7.4f}{r['mAP50']:>8.4f}{r['mAP50-95']:>10.4f}{r['FPS_FP16']:>7.1f}")

    print("\nPhan bo du doan tren valid (so box moi lop, conf TB):")
    for r in rows:
        dets = ", ".join(f"{l}={r[f'det_{l}']}({r[f'conf_{l}']:.2f})" for l in labels)
        print(f"  {r['model']}: {dets}")

    for r in rows:
        if r["per_class"]:
            print(f"\nmAP theo lop — {r['model']}:")
            for l, (a50, a95) in r["per_class"].items():
                print(f"  {l:>8}: mAP50={a50:.4f}  mAP50-95={a95:.4f}")

    # luu csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "precision", "recall", "map50", "map50_95", "fps_fp16"] +
                   [f"det_{l}" for l in labels])
        for r in rows:
            w.writerow([r["model"], r["P"], r["R"], r["mAP50"], r["mAP50-95"], r["FPS_FP16"]] +
                       [r[f"det_{l}"] for l in labels])
    print(f"\nDa luu: {OUT_CSV}")


if __name__ == "__main__":
    main()
