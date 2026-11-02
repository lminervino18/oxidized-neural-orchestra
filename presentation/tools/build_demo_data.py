"""Build demo/data.js: the demo architecture and the MNIST test set.

Usage (from presentation/):
    python3 tools/build_demo_data.py <mnist_dir>

<mnist_dir> holds mnist_test_samples.bin and mnist_test_labels.bin, packed
little-endian f32 as the orchestrator reads them (pixels in [0, 1], one-hot
labels). The page decompresses the set in the browser, so it works offline.
"""

import base64
import gzip
import json
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
X_SIZE = 784
Y_SIZE = 10


def read_f32(path):
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def main():
    src = Path(sys.argv[1])
    samples = read_f32(src / "mnist_test_samples.bin")
    labels = read_f32(src / "mnist_test_labels.bin")
    count = len(labels) // Y_SIZE
    assert len(samples) == count * X_SIZE

    images = bytes(round(v * 255) for v in samples)
    classes = bytes(labels[i * Y_SIZE:(i + 1) * Y_SIZE].index(1.0) for i in range(count))
    blob = base64.b64encode(gzip.compress(images + classes, 9)).decode()
    model = json.loads((ROOT / "demo" / "configs" / "model.json").read_text(encoding="utf-8"))

    payload = json.dumps({"model": model, "count": count, "test": blob})
    (ROOT / "demo" / "data.js").write_text(f"window.ONO_DEMO = {payload};\n", encoding="utf-8")
    print(f"demo/data.js: {count} test images, {len(blob) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
