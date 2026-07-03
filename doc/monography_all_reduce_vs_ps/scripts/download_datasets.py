#!/usr/bin/env python3
"""Ensure the monography datasets exist under datasets/.

MNIST: the full prepared .bin files already live at
``notebooks/mnist/data/mnist_{train,test}_{samples,labels}.bin`` (float32,
samples = 784 flattened normalized [0,1], labels = one-hot 10). We copy them into
``datasets/mnist/`` so the harness directory is self-contained.

EMNIST (optional, ``--emnist``): downloads the official NIST gzip bundle, extracts
the chosen split (default ``digits`` = 10 classes, matching the bundled models),
applies the EMNIST column-major transpose, normalizes /255 and writes the SAME
.bin format. ``balanced`` (47 classes) is also accepted but the bundled 10-output
models will not fit it without editing the final layer.

Usage:
    .venv/bin/python scripts/download_datasets.py                 # MNIST only
    .venv/bin/python scripts/download_datasets.py --emnist        # + EMNIST digits
    .venv/bin/python scripts/download_datasets.py --emnist --emnist-split balanced
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
MONO_DIR = SCRIPTS_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parents[2]

DATASETS_DIR = MONO_DIR / "datasets"
NB_MNIST_DIR = REPO_ROOT / "notebooks" / "mnist" / "data"

X_SIZE = 784

EMNIST_ZIP_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
EMNIST_CLASSES = {"digits": 10, "balanced": 47, "mnist": 10, "letters": 26}

# FashionMNIST — same IDX format and 10-class shape as MNIST (drop-in). Zalando's
# S3 website endpoint is the canonical source; the GitHub raw mirror is the fallback.
FASHION_MNIST_MIRRORS = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
]
FASHION_MNIST_FILES = {
    "train_s": "train-images-idx3-ubyte.gz",
    "train_l": "train-labels-idx1-ubyte.gz",
    "test_s": "t10k-images-idx3-ubyte.gz",
    "test_l": "t10k-labels-idx1-ubyte.gz",
}


# ── MNIST ─────────────────────────────────────────────────────────────────────

def ensure_mnist(symlink: bool = False) -> dict:
    src = {
        "train_s": NB_MNIST_DIR / "mnist_train_samples.bin",
        "train_l": NB_MNIST_DIR / "mnist_train_labels.bin",
        "test_s": NB_MNIST_DIR / "mnist_test_samples.bin",
        "test_l": NB_MNIST_DIR / "mnist_test_labels.bin",
    }
    missing = [str(p) for p in src.values() if not p.exists()]
    if missing:
        raise SystemExit(f"ERROR: source MNIST bins missing: {missing}")

    out = DATASETS_DIR / "mnist"
    out.mkdir(parents=True, exist_ok=True)
    dst = {
        "train_s": out / "mnist_train_samples.bin",
        "train_l": out / "mnist_train_labels.bin",
        "test_s": out / "mnist_test_samples.bin",
        "test_l": out / "mnist_test_labels.bin",
    }
    for key, s in src.items():
        d = dst[key]
        if d.exists() or d.is_symlink():
            d.unlink()
        if symlink:
            d.symlink_to(s.resolve())
        else:
            shutil.copyfile(s, d)

    n_train = dst["train_s"].stat().st_size // (X_SIZE * 4)
    n_test = dst["test_s"].stat().st_size // (X_SIZE * 4)
    print("== MNIST ==")
    print(f"  source        : {NB_MNIST_DIR} (LeCun MNIST, pre-converted)")
    print(f"  dest          : {out}")
    print(f"  sample shape  : 28x28 -> 784 flat, float32, normalized [0,1]")
    print(f"  label shape   : one-hot 10, float32")
    print(f"  classes       : 10 (digits 0-9)")
    print(f"  train / test  : {n_train} / {n_test}")
    print(f"  seed          : dataset order as-stored (deterministic)")
    return dst


# ── EMNIST (optional) ─────────────────────────────────────────────────────────

def _read_idx_images(raw: bytes, transpose: bool = True):
    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    assert magic == 2051, f"bad image magic {magic}"
    import numpy as np
    data = np.frombuffer(raw[16:16 + n * rows * cols], dtype=np.uint8)
    data = data.reshape(n, rows, cols)
    # EMNIST stores images column-major relative to MNIST orientation: transpose.
    # FashionMNIST uses the same row-major layout as MNIST, so it must NOT be transposed.
    if transpose:
        data = data.transpose(0, 2, 1)
    return data.reshape(n, rows * cols), rows, cols


def _read_idx_labels(raw: bytes):
    magic, n = struct.unpack(">II", raw[:8])
    assert magic == 2049, f"bad label magic {magic}"
    import numpy as np
    return np.frombuffer(raw[8:8 + n], dtype=np.uint8)


def _write_bins(images, labels, s_path: Path, l_path: Path, n_classes: int):
    import numpy as np
    x = (images.astype(np.float32) / 255.0)
    onehot = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    s_path.parent.mkdir(parents=True, exist_ok=True)
    x.tofile(str(s_path))
    onehot.tofile(str(l_path))


def ensure_emnist(split: str) -> dict:
    if split not in EMNIST_CLASSES:
        raise SystemExit(f"ERROR: unknown EMNIST split {split!r}; "
                         f"choose from {list(EMNIST_CLASSES)}")
    n_classes = EMNIST_CLASSES[split]
    out = DATASETS_DIR / "emnist"
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    raw_dir.mkdir(exist_ok=True)
    zip_path = raw_dir / "gzip.zip"

    if not zip_path.exists():
        print(f"  downloading EMNIST bundle from {EMNIST_ZIP_URL} (large, ~500MB)...")
        urllib.request.urlretrieve(EMNIST_ZIP_URL, zip_path)

    members = {
        "train_s": f"gzip/emnist-{split}-train-images-idx3-ubyte.gz",
        "train_l": f"gzip/emnist-{split}-train-labels-idx1-ubyte.gz",
        "test_s": f"gzip/emnist-{split}-test-images-idx3-ubyte.gz",
        "test_l": f"gzip/emnist-{split}-test-labels-idx1-ubyte.gz",
    }
    blobs = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for key, member in members.items():
            if member not in names:
                raise SystemExit(f"ERROR: {member} not in EMNIST zip "
                                 f"(available e.g. {sorted(n for n in names if split in n)[:4]})")
            with zf.open(member) as f:
                blobs[key] = gzip.decompress(f.read())

    train_x, rows, cols = _read_idx_images(blobs["train_s"])
    train_y = _read_idx_labels(blobs["train_l"])
    test_x, _, _ = _read_idx_images(blobs["test_s"])
    test_y = _read_idx_labels(blobs["test_l"])

    dst = {
        "train_s": out / "emnist_train_samples.bin",
        "train_l": out / "emnist_train_labels.bin",
        "test_s": out / "emnist_test_samples.bin",
        "test_l": out / "emnist_test_labels.bin",
    }
    _write_bins(train_x, train_y, dst["train_s"], dst["train_l"], n_classes)
    _write_bins(test_x, test_y, dst["test_s"], dst["test_l"], n_classes)

    print("== EMNIST ==")
    print(f"  source        : NIST EMNIST '{split}' split ({EMNIST_ZIP_URL})")
    print(f"  dest          : {out}")
    print(f"  sample shape  : {rows}x{cols} -> {rows*cols} flat, float32, /255, transposed")
    print(f"  label shape   : one-hot {n_classes}, float32")
    print(f"  classes       : {n_classes}")
    print(f"  train / test  : {train_x.shape[0]} / {test_x.shape[0]}")
    if n_classes != 10:
        print(f"  NOTE          : bundled models output 10 classes; '{split}' has "
              f"{n_classes} — adjust the final layer to use it.")
    return dst


# ── FashionMNIST (optional) ───────────────────────────────────────────────────

def _download_fashion(fname: str, dst: Path) -> None:
    last = None
    for base in FASHION_MNIST_MIRRORS:
        try:
            urllib.request.urlretrieve(base + fname, dst)
            return
        except Exception as e:  # noqa: BLE001 — try every mirror before giving up
            last = e
    raise SystemExit(f"ERROR: could not download {fname} from any mirror: {last}")


def ensure_fashion_mnist() -> dict:
    out = DATASETS_DIR / "fashion_mnist"
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    raw_dir.mkdir(exist_ok=True)

    blobs = {}
    for key, fname in FASHION_MNIST_FILES.items():
        gz = raw_dir / fname
        if not gz.exists():
            print(f"  downloading {fname}...")
            _download_fashion(fname, gz)
        blobs[key] = gzip.decompress(gz.read_bytes())

    # Row-major like MNIST: no transpose.
    train_x, rows, cols = _read_idx_images(blobs["train_s"], transpose=False)
    train_y = _read_idx_labels(blobs["train_l"])
    test_x, _, _ = _read_idx_images(blobs["test_s"], transpose=False)
    test_y = _read_idx_labels(blobs["test_l"])

    dst = {
        "train_s": out / "fashion_mnist_train_samples.bin",
        "train_l": out / "fashion_mnist_train_labels.bin",
        "test_s": out / "fashion_mnist_test_samples.bin",
        "test_l": out / "fashion_mnist_test_labels.bin",
    }
    _write_bins(train_x, train_y, dst["train_s"], dst["train_l"], 10)
    _write_bins(test_x, test_y, dst["test_s"], dst["test_l"], 10)

    print("== FashionMNIST ==")
    print(f"  source        : Zalando FashionMNIST ({FASHION_MNIST_MIRRORS[0]})")
    print(f"  dest          : {out}")
    print(f"  sample shape  : {rows}x{cols} -> {rows*cols} flat, float32, /255")
    print(f"  label shape   : one-hot 10, float32")
    print(f"  classes       : 10 (garments)")
    print(f"  train / test  : {train_x.shape[0]} / {test_x.shape[0]}")
    return dst


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare monography datasets.")
    ap.add_argument("--symlink", action="store_true",
                    help="symlink MNIST bins instead of copying")
    ap.add_argument("--emnist", action="store_true", help="also prepare EMNIST")
    ap.add_argument("--emnist-split", default="digits",
                    choices=list(EMNIST_CLASSES), help="EMNIST split (default digits)")
    ap.add_argument("--fashion", action="store_true",
                    help="also prepare FashionMNIST (drop-in 10-class replacement)")
    args = ap.parse_args()

    ensure_mnist(symlink=args.symlink)
    if args.emnist:
        ensure_emnist(args.emnist_split)
    if args.fashion:
        ensure_fashion_mnist()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
