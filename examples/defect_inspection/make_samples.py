#!/usr/bin/env python3
"""Generates synthetic brushed-metal surface images, some with defects.

Usage: make_samples.py OUT_DIR [--count 60] [--seed 7]

Writes OUT_DIR/<split>_<id>.png (8-bit gray, 960x540), one JSON Lines
manifest per split (train.jsonl, test.jsonl) for `framepipe batch`, and
labels.json mapping frame_id -> {"split", "label", "defects": [...]}.

Images are procedural so the example needs no third-party data. Defects are
thin dark cracks (random walks), straight scratches, and small pits: the
kinds of flaw an edge map makes visible. Half the images are defect-free.
The train split is used only to calibrate the detector threshold; accuracy
is reported on the held-out test split.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

W, H = 960, 540


def smooth1d(a, k, axis):
    """Box blur of width k along one axis (wrap-around), applied twice."""
    for _ in range(2):
        c = np.cumsum(np.concatenate([a, a.take(range(k), axis=axis)], axis=axis), axis=axis)
        a = (c.take(range(k, k + a.shape[axis]), axis=axis) - c.take(range(0, a.shape[axis]), axis=axis)) / k
    return a


def surface(rng):
    """Brushed metal: horizontal streaks, slow lighting gradient, sensor noise."""
    streaks = smooth1d(rng.normal(0, 1, (H, W)), 61, axis=1)
    streaks = streaks / (streaks.std() + 1e-9) * 7.0
    yy, xx = np.mgrid[0:H, 0:W]
    light = 18.0 * np.sin(xx / W * np.pi * rng.uniform(0.6, 1.4) + rng.uniform(0, np.pi)) * (yy / H - 0.5)
    noise = rng.normal(0, 4.0, (H, W))
    return 135.0 + rng.uniform(-15, 15) + streaks + light + noise


def draw_line(img, pts, depth, width):
    """Darkens pixels within `width` of the polyline by `depth` (anti-aliased)."""
    yy, xx = np.mgrid[0:H, 0:W]
    mask = np.zeros((H, W))
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        lo_x, hi_x = int(max(min(x0, x1) - 4, 0)), int(min(max(x0, x1) + 5, W))
        lo_y, hi_y = int(max(min(y0, y1) - 4, 0)), int(min(max(y0, y1) + 5, H))
        if lo_x >= hi_x or lo_y >= hi_y:
            continue
        px, py = xx[lo_y:hi_y, lo_x:hi_x], yy[lo_y:hi_y, lo_x:hi_x]
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
        d = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
        m = np.clip(1.0 - (d - width / 2), 0, 1)
        mask[lo_y:hi_y, lo_x:hi_x] = np.maximum(mask[lo_y:hi_y, lo_x:hi_x], m)
    img -= depth * mask
    ys, xs = np.nonzero(mask > 0.5)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None


def add_defect(img, rng):
    kind = rng.choice(["crack", "scratch", "pit"])
    if kind == "crack":
        x, y = rng.uniform(100, W - 100), rng.uniform(80, H - 80)
        ang = rng.uniform(0, 2 * np.pi)
        pts = [(x, y)]
        for _ in range(int(rng.integers(25, 50))):
            ang += rng.normal(0, 0.35)
            x, y = x + 4 * np.cos(ang), y + 4 * np.sin(ang)
            pts.append((float(np.clip(x, 2, W - 3)), float(np.clip(y, 2, H - 3))))
        box = draw_line(img, pts, rng.uniform(35, 60), rng.uniform(1.0, 2.0))
    elif kind == "scratch":
        x0, y0 = rng.uniform(50, W - 50), rng.uniform(50, H - 50)
        ang, length = rng.uniform(0, np.pi), rng.uniform(120, 320)
        x1 = float(np.clip(x0 + length * np.cos(ang), 2, W - 3))
        y1 = float(np.clip(y0 + length * np.sin(ang), 2, H - 3))
        box = draw_line(img, [(x0, y0), (x1, y1)], rng.uniform(25, 45), rng.uniform(0.8, 1.5))
    else:
        cx, cy, r = rng.uniform(60, W - 60), rng.uniform(60, H - 60), rng.uniform(4, 9)
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.hypot(xx - cx, yy - cy)
        img -= rng.uniform(40, 70) * np.clip(r - d + 1, 0, 1)
        box = [int(cx - r), int(cy - r), int(cx + r), int(cy + r)]
    return {"kind": str(kind), "box": box}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=60, help="images per split (default 60)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    labels = {}
    for split in ("train", "test"):
        with open(os.path.join(a.out, f"{split}.jsonl"), "w") as man:
            for i in range(a.count):
                fid = f"{split}_{i:03d}"
                img = surface(rng)
                defects = []
                if i % 2 == 1:  # half the images carry one or two defects
                    for _ in range(int(rng.integers(1, 3))):
                        defects.append(add_defect(img, rng))
                Image.fromarray(np.clip(img, 0, 255).round().astype(np.uint8), "L").save(
                    os.path.join(a.out, fid + ".png"))
                man.write(json.dumps({"frame_id": fid, "path": fid + ".png"}) + "\n")
                labels[fid] = {"split": split, "label": "defect" if defects else "good", "defects": defects}
    with open(os.path.join(a.out, "labels.json"), "w") as f:
        json.dump({"seed": a.seed, "width": W, "height": H, "images": labels}, f, indent=1)
    n_def = sum(v["label"] == "defect" for v in labels.values())
    print(f"make_samples: {len(labels)} images ({n_def} with defects) in {a.out}")


if __name__ == "__main__":
    main()
