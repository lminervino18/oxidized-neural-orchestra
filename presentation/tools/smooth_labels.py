"""Write a label-smoothed copy of a one-hot labels file for the live demo.

Usage (from the repository root):
    python3 presentation/tools/smooth_labels.py <labels.bin> <out.bin> [epsilon]

Each label becomes (1 - epsilon) * one_hot + epsilon / classes (Szegedy et al.,
2016). With a positive floor the training loss flattens once the model has
learned, so the Strategy-Switch trigger (mean relative loss change under 1 %
over 6 rounds) fires reliably during the demo. Accuracy is still measured
against the original labels.
"""

import sys
from array import array
from pathlib import Path

CLASSES = 10


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    labels = array("f")
    labels.frombytes(src.read_bytes())
    smoothed = array("f", ((1 - epsilon) * v + epsilon / CLASSES for v in labels))
    dst.write_bytes(smoothed.tobytes())
    print(f"{dst}: {len(labels) // CLASSES} labels, epsilon {epsilon}")


if __name__ == "__main__":
    main()
