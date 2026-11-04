#!/usr/bin/env python3
"""Independent NumPy oracle for the FramePipe numerical contract (spec Section 7).

Written directly from the specification. It does NOT import, bind to, or call
any FramePipe C++ code, and it recomputes resize coordinates and Gaussian
weights itself in double precision (it does not read production tables).

All arithmetic is float64. Neighborhood operations use explicit index arrays
for replicated borders (no library filter defaults).

Library use:
    import oracle
    stages = oracle.run_pipeline(img_u8, out_w, out_h, radius, sigma, "edges")
    stages["edges"]  # float64 array (H, W)

Command-line use (writes <stage>.f32 files that `framepipe validate
--reference oracle:DIR` reads):
    python3 scripts/oracle.py --input in.png --width 64 --height 48 \
        --radius 3 --sigma 1.2 --output-stage edges --out-dir DIR
The CLI decodes images with a minimal pure-Python PNG reader (8-bit gray/RGB,
non-interlaced) so it is independent of OpenCV as well.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import zlib

try:
    import numpy as np
except ImportError:  # pragma: no cover - reported by callers
    np = None

STAGES = ["gray", "resize", "blur_horizontal", "blur", "sobel_x", "sobel_y", "edges"]


def _clamp_idx(i, n):
    return np.minimum(np.maximum(i, 0), n - 1)


def grayscale_u8(img: "np.ndarray") -> "np.ndarray":
    """Spec 7.2. img is uint8 (H, W) gray or (H, W, 3) RGB."""
    if img.ndim == 2:
        return img.astype(np.int64)
    r = img[..., 0].astype(np.int64)
    g = img[..., 1].astype(np.int64)
    b = img[..., 2].astype(np.int64)
    return (77 * r + 150 * g + 29 * b + 128) >> 8


def grayscale(img):
    return grayscale_u8(img).astype(np.float64) / 255.0


def resize_bilinear(src: "np.ndarray", wo: int, ho: int) -> "np.ndarray":
    """Spec 7.3: half-pixel centers, floor, weight from unclamped index, clamp both samples."""
    hi, wi = src.shape
    xs = np.arange(wo, dtype=np.float64)
    ys = np.arange(ho, dtype=np.float64)
    u = (xs + 0.5) * float(wi) / float(wo) - 0.5
    v = (ys + 0.5) * float(hi) / float(ho) - 0.5
    x0r = np.floor(u)
    y0r = np.floor(v)
    ax = u - x0r
    ay = v - y0r
    x0 = _clamp_idx(x0r.astype(np.int64), wi)
    x1 = _clamp_idx(x0r.astype(np.int64) + 1, wi)
    y0 = _clamp_idx(y0r.astype(np.int64), hi)
    y1 = _clamp_idx(y0r.astype(np.int64) + 1, hi)
    p00 = src[np.ix_(y0, x0)]
    p01 = src[np.ix_(y0, x1)]
    p10 = src[np.ix_(y1, x0)]
    p11 = src[np.ix_(y1, x1)]
    axb = ax[None, :]
    ayb = ay[:, None]
    top = p00 + (p01 - p00) * axb
    bottom = p10 + (p11 - p10) * axb
    return top + (bottom - top) * ayb


def gaussian_weights(radius: int, sigma: float) -> "np.ndarray":
    """Spec 7.4, double precision. Returned in float64 (oracle never rounds to FP32)."""
    if not (0 <= radius <= 15) or not math.isfinite(sigma) or not (0.1 <= sigma <= 100.0):
        raise ValueError("gaussian parameters out of range")
    ks = range(-radius, radius + 1)
    un = [math.exp(-(float(k) * float(k)) / (2.0 * sigma * sigma)) for k in ks]
    s = math.fsum(un)
    return np.array([x / s for x in un], dtype=np.float64)


def gaussian_horizontal(src, w, radius):
    h, wd = src.shape
    xs = np.arange(wd)
    out = np.zeros_like(src)
    for i, k in enumerate(range(-radius, radius + 1)):
        out += w[i] * src[:, _clamp_idx(xs + k, wd)]
    return out


def gaussian_vertical(src, w, radius):
    h, wd = src.shape
    ys = np.arange(h)
    out = np.zeros_like(src)
    for i, k in enumerate(range(-radius, radius + 1)):
        out += w[i] * src[_clamp_idx(ys + k, h), :]
    return out


GX = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
GY = ((-1, -2, -1), (0, 0, 0), (1, 2, 1))


def sobel(src):
    """Spec 7.5: correlation (no kernel flip), replicated borders."""
    h, w = src.shape
    xs = np.arange(w)
    ys = np.arange(h)
    gx = np.zeros_like(src)
    gy = np.zeros_like(src)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sample = src[np.ix_(_clamp_idx(ys + dy, h), _clamp_idx(xs + dx, w))]
            cx = GX[dy + 1][dx + 1]
            cy = GY[dy + 1][dx + 1]
            if cx:
                gx += cx * sample
            if cy:
                gy += cy * sample
    mag = np.sqrt(gx * gx + gy * gy)
    edges = np.clip(mag / math.sqrt(32.0), 0.0, 1.0)
    return gx, gy, edges


def run_pipeline(img, wo, ho, radius=3, sigma=1.2, output_stage="edges"):
    """Returns dict of stage name -> float64 array for the executed prefix."""
    out = {}
    out["gray"] = grayscale(img)
    out["resize"] = resize_bilinear(out["gray"], wo, ho)
    if output_stage == "resize":
        return out
    w = gaussian_weights(radius, sigma)
    out["blur_horizontal"] = gaussian_horizontal(out["resize"], w, radius)
    out["blur"] = gaussian_vertical(out["blur_horizontal"], w, radius)
    if output_stage == "blur":
        return out
    gx, gy, e = sobel(out["blur"])
    out["sobel_x"] = gx
    out["sobel_y"] = gy
    out["edges"] = e
    return out


def quantize(v):
    """Spec 7.6."""
    return np.floor(np.clip(v, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- PNG I/O
# Minimal, dependency-free PNG codec for 8-bit gray (color type 0) and RGB
# (color type 2), non-interlaced. Used by the oracle CLI and fixture generator.

def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos = 8
    idat = b""
    w = h = depth = ctype = interlace = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    if depth != 8 or ctype not in (0, 2) or interlace != 0:
        raise ValueError(f"{path}: oracle PNG reader supports only 8-bit gray/RGB non-interlaced")
    ch = 1 if ctype == 0 else 3
    raw = zlib.decompress(idat)
    stride = w * ch
    out = bytearray(h * stride)
    prev = bytearray(stride)
    p = 0
    for y in range(h):
        ft = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        for i in range(stride):
            a = line[i - ch] if i >= ch else 0
            b = prev[i]
            c = prev[i - ch] if i >= ch else 0
            if ft == 1:
                line[i] = (line[i] + a) & 0xFF
            elif ft == 2:
                line[i] = (line[i] + b) & 0xFF
            elif ft == 3:
                line[i] = (line[i] + ((a + b) >> 1)) & 0xFF
            elif ft == 4:
                line[i] = (line[i] + _paeth(a, b, c)) & 0xFF
            elif ft != 0:
                raise ValueError("bad PNG filter")
        out[y * stride:(y + 1) * stride] = line
        prev = line
    arr = np.frombuffer(bytes(out), dtype=np.uint8)
    return arr.reshape(h, w) if ch == 1 else arr.reshape(h, w, 3)


def write_png(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        h, w = arr.shape
        ctype = 0
    else:
        h, w, _ = arr.shape
        ctype = 2
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def write_f32(path, arr):
    """FPF32 format read by `framepipe validate --reference oracle:DIR`.
    The float64 oracle value is rounded to the nearest float32 only for storage."""
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(f"FPF32 {w} {h}\n".encode("ascii"))
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def main(argv=None):
    if np is None:
        print("oracle.py requires NumPy (pip install numpy==2.1.3)", file=sys.stderr)
        return 77
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--output-stage", choices=["resize", "blur", "edges"], default="edges")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    img = read_png(a.input)
    st = run_pipeline(img, a.width, a.height, a.radius, a.sigma, a.output_stage)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, arr in st.items():
        write_f32(os.path.join(a.out_dir, name + ".f32"), arr)
    print(f"oracle: wrote {len(st)} stages to {a.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
