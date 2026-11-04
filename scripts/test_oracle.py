#!/usr/bin/env python3
"""Self-tests for the independent oracle against the hand-checkable fixtures
in spec Section 16.4. Exit 77 (ctest SKIP) if NumPy is missing."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
except ImportError:
    print("SKIP: NumPy not installed", file=sys.stderr)
    sys.exit(77)

import oracle  # noqa: E402

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("FAIL:", msg, file=sys.stderr)


# 16.4.1 Gray 1x1 = 128
img = np.array([[128]], dtype=np.uint8)
for w, h in [(1, 1), (5, 3), (16, 16)]:
    st = oracle.run_pipeline(img, w, h, 3, 1.2, "edges")
    check(np.allclose(st["resize"], 128 / 255, atol=1e-12), "1x1 resize")
    check(np.allclose(st["blur"], 128 / 255, atol=1e-12), "1x1 blur")
    check(np.all(st["edges"] == 0), "1x1 edges zero")

# 16.4.2 primary colors
rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
check(list(oracle.grayscale_u8(rgb)[0]) == [77, 149, 29, 255, 0], "primary colors")

# 16.4.3 2x2 -> 3x3 and 1x1
src = np.array([[0.0, 1.0], [1.0, 0.0]])
r3 = oracle.resize_bilinear(src, 3, 3)
check(np.allclose(r3, [[0, 0.5, 1], [0.5, 0.5, 0.5], [1, 0.5, 0]], atol=1e-12), f"2x2->3x3 got {r3}")
check(abs(oracle.resize_bilinear(src, 1, 1)[0, 0] - 0.5) < 1e-12, "2x2->1x1")

# 16.4.4 Sobel on a horizontal ramp
for W in (3, 4, 9):
    p = np.tile(np.arange(W, dtype=np.float64) / (W - 1), (5, 1))
    gx, gy, e = oracle.sobel(p)
    check(np.allclose(gx[:, 1:-1], 8 / (W - 1)), f"ramp gx W={W}")
    check(np.allclose(gy, 0), f"ramp gy W={W}")
    check(np.allclose(e[:, 1:-1], math.sqrt(2) / (W - 1)), f"ramp edge W={W}")

# 16.4.5 constant stays constant for any radius
for r in (0, 1, 3, 15):
    w = oracle.gaussian_weights(r, 2.0)
    c = np.full((7, 5), 0.3)
    out = oracle.gaussian_vertical(oracle.gaussian_horizontal(c, w, r), w, r)
    check(np.allclose(out, 0.3, atol=1e-12), f"constant radius {r}")

# Weight normalization and range checks
check(abs(oracle.gaussian_weights(15, 0.1).sum() - 1) < 1e-12, "weights normalized")
for bad in [(-1, 1.0), (16, 1.0), (3, 0.05), (3, float("nan"))]:
    try:
        oracle.gaussian_weights(*bad)
        check(False, f"weights accepted {bad}")
    except ValueError:
        pass

# Quantization
check(list(oracle.quantize(np.array([0.0, 1.0, 0.5, -1, 2]))) == [0, 255, 128, 0, 255], "quantize")

# PNG round trip through the pure-Python codec
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    a = (np.arange(7 * 5 * 3) % 256).astype(np.uint8).reshape(5, 7, 3)
    oracle.write_png(os.path.join(d, "a.png"), a)
    check(np.array_equal(oracle.read_png(os.path.join(d, "a.png")), a), "png rgb roundtrip")
    g = (np.arange(13 * 4) * 5 % 256).astype(np.uint8).reshape(4, 13)
    oracle.write_png(os.path.join(d, "g.png"), g)
    check(np.array_equal(oracle.read_png(os.path.join(d, "g.png")), g), "png gray roundtrip")

if failures:
    print(f"{len(failures)} oracle self-test failure(s)", file=sys.stderr)
    sys.exit(1)
print("oracle self-tests passed", file=sys.stderr)
