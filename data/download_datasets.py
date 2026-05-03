"""
Download, merge, and organize fabric defect datasets into:
  images/train/{stain,tear,weave_distortion,normal}/
  images/val/{stain,tear,weave_distortion,normal}/

Usage:
  python data/download_datasets.py
  python data/download_datasets.py --val-split 0.2 --out data/images

Dependencies:
  pip install kagglehub roboflow python-dotenv tqdm pillow

Kaggle credentials must be set up via:
  ~/.kaggle/kaggle.json  OR  env vars KAGGLE_USERNAME + KAGGLE_KEY
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Dataset slugs (verified)
# ---------------------------------------------------------------------------

# Primary: ~2.7 k images — hole, horizontal, line, stain, vertical defects + defect-free
# kaggle.com/datasets/rmshashi/fabric-defect-dataset
KAGGLE_PRIMARY = "rmshashi/fabric-defect-dataset"

# Stain focus: 398 stain images (ink, oil, dirt) + 68 clean
# kaggle.com/datasets/priemshpathirana/fabric-stain-dataset
KAGGLE_STAIN = "priemshpathirana/fabric-stain-dataset"

# Multi-class defect dataset (weave, holes, pulls, stains)
# kaggle.com/datasets/nexuswho/fabric-defects-dataset
KAGGLE_EXTRA = "nexuswho/fabric-defects-dataset"

# Roboflow "Fabric Defect Detection" — public dataset, 4 960 images
# universe.roboflow.com/fyp-bgzyz/fabric-defect-detection-rcota
ROBOFLOW_WORKSPACE = "fyp-bgzyz"
ROBOFLOW_PROJECT   = "fabric-defect-detection-rcota"
ROBOFLOW_VERSION   = 1
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY", "")   # set in .env

# ---------------------------------------------------------------------------
# Class mapping: folder / filename keywords → target class
# Order matters — first match wins; longer/more-specific terms first
# ---------------------------------------------------------------------------

CLASS_MAP: dict[str, list[str]] = {
    "stain": [
        "oil_stain", "dye_stain", "colour_bleed", "color_bleed",
        "ink_stain", "dirt_stain", "stain_defect",
        "stain", "stains", "dirty", "contamination", "spot",
    ],
    "tear": [
        "hole_snag", "broken_thread", "broken_end", "hole_defect",
        "hole", "holes", "snag", "tear", "tears", "cut",
        "damage", "rip", "missing_yarn", "broken",
    ],
    "weave_distortion": [
        "weave_distortion", "drop_stitch", "shade_variation",
        "horizontal_defect", "vertical_defect", "line_defect",
        "horizontal", "vertical", "verticle", "lines", "line",
        "weave", "distortion", "slub", "nep", "run",
        "shrinkage", "skew", "bow", "thread_error", "pull",
    ],
    "normal": [
        "no_defect", "defect_free", "defect free", "good_fabric",
        "normal", "clean", "good", "background", "negative", "ok",
    ],
}

TARGET_CLASSES = list(CLASS_MAP.keys())
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify(path: Path) -> str | None:
    """Return the target class for an image by checking only its nearest 3 ancestor folders.

    Avoids false matches from dataset-level folder names (e.g. 'fabric-stain-dataset'
    containing the word 'stain', which would incorrectly classify all images in it).
    """
    # Normalise: lowercase + collapse spaces/hyphens to underscores
    def norm(s: str) -> str:
        return s.lower().replace(" ", "_").replace("-", "_")

    # Only check immediate parent, grandparent, and great-grandparent
    ancestors = [norm(p) for p in path.parts[-4:-1]]
    for target_cls, keywords in CLASS_MAP.items():
        for ancestor in ancestors:
            for kw in keywords:
                if kw in ancestor:
                    return target_cls
    return None


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]


def copy_images(
    src_images: list[Path],
    dest_root: Path,
    split: str,
    source_tag: str,
    counters: dict,
) -> None:
    for img_path in tqdm(src_images, desc=f"  copying {source_tag}/{split}", unit="img", leave=False):
        cls = classify(img_path)
        if cls is None:
            counters["skipped"] += 1
            continue
        if not is_valid_image(img_path):
            counters["corrupt"] += 1
            continue

        dest_dir = dest_root / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename: <source_tag>_<original_stem>_<counter><ext>
        counters[cls] += 1
        dest_name = f"{source_tag}_{img_path.stem}_{counters[cls]}{img_path.suffix.lower()}"
        dest_path = dest_dir / dest_name

        if not dest_path.exists():
            shutil.copy2(img_path, dest_path)


def train_val_split(images: list[Path], val_ratio: float, seed: int = 42) -> tuple[list, list]:
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

def download_kaggle(slug: str, label: str) -> Path | None:
    """Download a Kaggle dataset with kagglehub; return local cache path."""
    try:
        import kagglehub
    except ImportError:
        print("  [ERROR] kagglehub not installed. Run: pip install kagglehub")
        return None

    print(f"  Downloading {label} ({slug}) …")
    try:
        path = kagglehub.dataset_download(slug)
        print(f"  Cached at: {path}")
        return Path(path)
    except Exception as exc:
        print(f"  [WARN] Could not download {slug}: {exc}")
        print(f"         Check the slug at: https://www.kaggle.com/datasets/{slug}")
        return None


def download_roboflow(out_dir: Path) -> Path | None:
    """Export the Roboflow dataset in YOLOv8 format and return its folder."""
    if not ROBOFLOW_API_KEY:
        print("  [SKIP] ROBOFLOW_API_KEY not set in .env — skipping Roboflow dataset.")
        return None
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  [ERROR] roboflow not installed. Run: pip install roboflow")
        return None

    print(f"  Downloading Roboflow: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION} …")
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        dataset = project.version(ROBOFLOW_VERSION).download("yolov8", location=str(out_dir / "roboflow_raw"))
        path = Path(dataset.location)
        print(f"  Roboflow data at: {path}")
        return path
    except Exception as exc:
        print(f"  [WARN] Roboflow download failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Per-dataset ingestion  (handles varied folder layouts)
# ---------------------------------------------------------------------------

def ingest(
    raw_root: Path,
    dest_root: Path,
    source_tag: str,
    val_ratio: float,
    counters: dict,
) -> None:
    if raw_root is None:
        return
    all_images = collect_images(raw_root)
    if not all_images:
        print(f"  [WARN] No images found under {raw_root}")
        return

    print(f"  Found {len(all_images)} images in {source_tag}")
    train_imgs, val_imgs = train_val_split(all_images, val_ratio)
    copy_images(train_imgs, dest_root, "train", source_tag, counters)
    copy_images(val_imgs, dest_root, "val", source_tag, counters)


def ingest_yolo(
    yolo_root: Path,
    dest_root: Path,
    source_tag: str,
    counters: dict,
) -> None:
    """Ingest a YOLOv8-format dataset (train/images + train/labels) by mapping
    class names from data.yaml through CLASS_MAP into the target folders.

    The YOLO dataset already has its own train/valid split, so we respect it.
    """
    if yolo_root is None:
        return

    # Parse class names from data.yaml
    data_yaml = yolo_root / "data.yaml"
    if not data_yaml.exists():
        print(f"  [WARN] No data.yaml found in {yolo_root} — skipping YOLO ingest")
        return

    import yaml as _yaml
    with open(data_yaml) as f:
        meta = _yaml.safe_load(f)
    yolo_classes: list[str] = meta.get("names", [])
    print(f"  YOLO classes: {yolo_classes}")

    # Pre-build class_id → target_class lookup
    id_to_target: dict[int, str] = {}
    for i, name in enumerate(yolo_classes):
        norm_name = name.lower().replace(" ", "_").replace("-", "_")
        for target_cls, keywords in CLASS_MAP.items():
            if any(kw in norm_name for kw in keywords):
                id_to_target[i] = target_cls
                break

    print(f"  Class mapping: { {yolo_classes[i]: v for i, v in id_to_target.items()} }")

    for split_dir_name, dest_split in [("train", "train"), ("valid", "val"), ("test", "val")]:
        img_dir = yolo_root / split_dir_name / "images"
        lbl_dir = yolo_root / split_dir_name / "labels"
        if not img_dir.exists():
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        print(f"  Found {len(images)} images in Roboflow/{split_dir_name}")

        for img_path in tqdm(images, desc=f"  copying roboflow/{dest_split}", unit="img", leave=False):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                counters["skipped"] += 1
                continue

            # Read all class IDs in the label file; use the most frequent one
            class_ids = []
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        class_ids.append(int(parts[0]))
                    except ValueError:
                        pass

            if not class_ids:
                counters["skipped"] += 1
                continue

            # Pick the most common class in this image
            dominant_id = max(set(class_ids), key=class_ids.count)
            target_cls = id_to_target.get(dominant_id)
            if target_cls is None:
                counters["skipped"] += 1
                continue

            if not is_valid_image(img_path):
                counters["corrupt"] += 1
                continue

            dest_dir = dest_root / dest_split / target_cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            counters[target_cls] += 1
            dest_name = f"{source_tag}_{img_path.stem}_{counters[target_cls]}{img_path.suffix.lower()}"
            dest_path = dest_dir / dest_name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir: Path, val_ratio: float) -> None:
    print(f"\n=== Fabric Defect Dataset Builder ===")
    print(f"Output  : {out_dir}")
    print(f"Val split: {val_ratio:.0%}\n")

    counters: dict = defaultdict(int)
    rf_raw_dir = out_dir.parent / "raw_downloads"

    # 1. Primary Kaggle dataset (~2.7k images: holes, horizontal, line, stain, vertical, defect-free)
    print("[1/4] Primary dataset — rmshashi/fabric-defect-dataset")
    primary_path = download_kaggle(KAGGLE_PRIMARY, "Fabric Defect Dataset")
    ingest(primary_path, out_dir, "primary", val_ratio, counters)

    # 2. Stain-focused dataset (398 stain images + 68 clean)
    print("\n[2/4] Stain dataset — priemshpathirana/fabric-stain-dataset")
    stain_path = download_kaggle(KAGGLE_STAIN, "Fabric Stain Dataset")
    ingest(stain_path, out_dir, "stain", val_ratio, counters)

    # 3. Extra multi-class dataset (nexuswho/fabric-defects-dataset)
    print("\n[3/4] Extra dataset — nexuswho/fabric-defects-dataset")
    extra_path = download_kaggle(KAGGLE_EXTRA, "Fabric Defects Dataset")
    ingest(extra_path, out_dir, "extra", val_ratio, counters)

    # 4. Roboflow Fabric Defect Detection (YOLOv8 format — has its own train/valid split)
    print("\n[4/4] Roboflow — Fabric Defect Detection")
    rf_path = download_roboflow(rf_raw_dir)
    ingest_yolo(rf_path, out_dir, "roboflow", counters)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n=== Dataset Summary ===")
    total = 0
    for split in ("train", "val"):
        print(f"\n  {split}/")
        for cls in TARGET_CLASSES:
            cls_dir = out_dir / split / cls
            n = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
            total += n
            bar = "█" * (n // 20) if n else "-"
            print(f"    {cls:<18} {n:>5} images  {bar}")

    print(f"\n  Total images copied : {total}")
    print(f"  Skipped (unmapped)  : {counters['skipped']}")
    print(f"  Corrupt / unreadable: {counters['corrupt']}")

    if counters["skipped"] > 0:
        print("\n  TIP: Review CLASS_MAP in this script to capture more of the skipped images.")

    print(f"\nDone. Dataset ready at: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and organize fabric defect datasets.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "images",
        help="Destination root (default: data/images/)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of images reserved for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    args = parser.parse_args()

    if not (0.0 < args.val_split < 1.0):
        sys.exit("--val-split must be between 0 and 1 (exclusive)")

    random.seed(args.seed)
    main(args.out.resolve(), args.val_split)
