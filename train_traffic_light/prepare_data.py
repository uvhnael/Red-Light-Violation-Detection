#!/usr/bin/env python3
"""Prepare the LISA cropped traffic-light dataset for 3-class training.

Source: Kaggle dataset ``chandanakuntala/cropped-lisa-traffic-light-dataset``
(downloaded via kagglehub, cached under ~/.cache/kagglehub).

The original dataset has 7 fine-grained classes.  For the RLVD pipeline we
only need the lamp colour, so they are merged:

    red    <- stop, stopLeft
    yellow <- warning, warningLeft
    green  <- go, goForward, goLeft

Output layout (Ultralytics classification format)::

    dataset/
      train/{red,yellow,green}/*.jpg
      val/{red,yellow,green}/*.jpg
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

# original LISA class -> merged colour class
CLASS_MAP = {
    "stop": "red",
    "stopLeft": "red",
    "warning": "yellow",
    "warningLeft": "yellow",
    "go": "green",
    "goForward": "green",
    "goLeft": "green",
}


def download_source() -> Path:
    """Return the path to the extracted kagglehub dataset (download if needed)."""
    import kagglehub

    path = Path(
        kagglehub.dataset_download("chandanakuntala/cropped-lisa-traffic-light-dataset")
    )
    root = path / "cropped_lisa_1"
    if not root.is_dir():
        raise RuntimeError(f"Unexpected dataset layout under {path}")
    return root


def build_split(src_root: Path, split_src: str, split_dst: str) -> dict[str, int]:
    """Copy images from src_root/<split_src>/<class>/ into dataset/<split_dst>/<colour>/."""
    counts: dict[str, int] = {}
    src_split = src_root / split_src
    for cls_dir in sorted(src_split.iterdir()):
        if not cls_dir.is_dir():
            continue
        colour = CLASS_MAP.get(cls_dir.name)
        if colour is None:
            print(f"  WARNING: unknown class {cls_dir.name!r} — skipped")
            continue
        dst_dir = DATASET_DIR / split_dst / colour
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in cls_dir.glob("*.jpg"):
            shutil.copy2(img, dst_dir / img.name)
            n += 1
        counts[colour] = counts.get(colour, 0) + n
        print(f"  {split_dst}/{colour}: +{n} from {cls_dir.name}")
    return counts


def main() -> int:
    if DATASET_DIR.exists():
        print(f"Removing existing dataset at {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    print("Locating/downloading source dataset ...")
    src_root = download_source()
    print(f"Source: {src_root}")

    print("\nBuilding train split:")
    train_counts = build_split(src_root, "train_1", "train")
    print("\nBuilding val split:")
    val_counts = build_split(src_root, "val_1", "val")

    print("\nSummary:")
    for colour in ("red", "yellow", "green"):
        print(f"  {colour:6s} train={train_counts.get(colour, 0):6d}  val={val_counts.get(colour, 0):5d}")
    print(f"\nDataset ready at {DATASET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
