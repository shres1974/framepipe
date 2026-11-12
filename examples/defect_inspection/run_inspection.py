#!/usr/bin/env python3
"""Surface-defect inspection built on FramePipe's edge maps.

Usage:
  run_inspection.py --framepipe build/cpu/bin/framepipe --samples DIR --out DIR [--backend auto]

1. Runs the real `framepipe batch` CLI on the train and test manifests from
   make_samples.py (grayscale -> resize -> blur -> Sobel edges).
2. Calibrates on the TRAIN split only: a grid search over the edge threshold
   (in 8-bit levels, because outputs are quantized PNGs) and the minimum
   region area picks the rule with the best train accuracy.
3. Classifies each held-out TEST image: pixels above the threshold are
   grouped into 8-connected regions, and the image is a "defect" if any
   region has at least min_area pixels. The regions give bounding boxes.
4. Writes report.json (threshold, confusion matrix, precision/recall,
   localization hit rate, FramePipe throughput) and prints a summary.

This is a deliberately simple, explainable detector. It shows where an edge
pipeline fits in a quality-control system; it is not a production
inspection model.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import deque

import numpy as np
from PIL import Image


def run_batch(fp, manifest, out_dir, backend, width, height):
    cmd = [fp, "batch", "--manifest", manifest, "--output-dir", out_dir, "--backend", backend,
           "--width", str(width), "--height", str(height), "--overwrite"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.load(open(os.path.join(out_dir, "batch_report.json")))


def load_edges(path):
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def regions(mask):
    """8-connected components of a boolean mask -> list of (area, [x0, y0, x1, y1])."""
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        area, xs, ys = 0, [], []
        while q:
            y, x = q.popleft()
            area += 1
            xs.append(x)
            ys.append(y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        out.append((area, [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]))
    return out


def detect(edges, threshold, min_area):
    found = [r for r in regions(edges > threshold) if r[0] >= min_area]
    return [{"area": a, "box": b} for a, b in sorted(found, reverse=True)]


def overlaps(a, b, pad=4):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--framepipe", required=True)
    ap.add_argument("--samples", required=True, help="directory written by make_samples.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--min-area-grid", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3, 4, 6, 8, 12],
                    help="candidate minimum region areas for calibration (default 1,2,3,4,6,8,12)")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.samples, "labels.json")))
    labels, W, H = meta["images"], meta["width"], meta["height"]
    os.makedirs(a.out, exist_ok=True)

    reports = {}
    for split in ("train", "test"):
        reports[split] = run_batch(a.framepipe, os.path.join(a.samples, f"{split}.jsonl"),
                                   os.path.join(a.out, f"edges_{split}"), a.backend, W, H)

    # Calibrate on the TRAIN split only: grid-search the threshold (in 8-bit
    # edge levels, since outputs are quantized PNGs) and the minimum region
    # area for the best train accuracy. Ties prefer fewer false positives,
    # then the higher threshold and larger area (more margin).
    train = [(load_edges(os.path.join(a.out, "edges_train", f + ".png")), v["label"] == "defect")
             for f, v in sorted(labels.items()) if v["split"] == "train"]
    good_max = max(float(e.max()) for e, is_def in train if not is_def)
    base = int(round(good_max * 255))
    best = None
    for level in range(max(base - 3, 1), base + 8):
        t = (level + 0.5) / 255.0
        areas = [(max([r[0] for r in regions(e > t)] or [0]), is_def) for e, is_def in train]
        for min_area in a.min_area_grid:
            tp_ = sum(ar >= min_area and d for ar, d in areas)
            fp_ = sum(ar >= min_area and not d for ar, d in areas)
            acc = (tp_ + sum(ar < min_area and not d for ar, d in areas)) / len(areas)
            key = (acc, -fp_, level, min_area)
            if best is None or key > best[0]:
                best = (key, t, min_area, acc, level)
    _, threshold, min_area, train_acc, level = best

    results, tp, fp, tn, fn, loc_hits, loc_total = {}, 0, 0, 0, 0, 0, 0
    for fid in sorted(f for f, v in labels.items() if v["split"] == "test"):
        truth = labels[fid]
        found = detect(load_edges(os.path.join(a.out, "edges_test", fid + ".png")), threshold, min_area)
        pred = "defect" if found else "good"
        if truth["label"] == "defect":
            tp, fn = tp + (pred == "defect"), fn + (pred == "good")
            for d in truth["defects"]:
                loc_total += 1
                loc_hits += any(overlaps(d["box"], f["box"]) for f in found)
        else:
            fp, tn = fp + (pred == "defect"), tn + (pred == "good")
        results[fid] = {"truth": truth["label"], "predicted": pred, "regions": found, "defects": truth["defects"]}

    n = tp + fp + tn + fn
    test_batch = reports["test"]
    report = {
        "detector": {"rule": "region of >= min_area 8-connected pixels with edge > threshold",
                     "threshold": threshold, "threshold_level_8bit": level, "min_area": min_area,
                     "calibration": "grid search on the train split only (threshold level x min_area)",
                     "train_accuracy": train_acc, "train_good_max_edge": good_max},
        "test": {"images": n, "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                 "accuracy": (tp + tn) / n if n else None,
                 "precision": tp / (tp + fp) if tp + fp else None,
                 "recall": tp / (tp + fn) if tp + fn else None,
                 "localization_hit_rate": loc_hits / loc_total if loc_total else None},
        "framepipe": {"backend": test_batch["selected_backend"], "implementation": test_batch["implementation"],
                      "batch_fps": test_batch["throughput_completed_fps"], "policy": test_batch["policy"]},
        "images": results,
    }
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    t = report["test"]
    print(f"inspection: {n} held-out test images; threshold edge level > {level}, min area {min_area} px "
          f"(calibrated on train, train accuracy {train_acc:.1%})")
    print(f"  confusion   TP {tp}  FP {fp}  TN {tn}  FN {fn}")
    print(f"  accuracy {t['accuracy']:.1%}  precision {t['precision'] or 0:.1%}  recall {t['recall'] or 0:.1%}  "
          f"defects localized {loc_hits}/{loc_total}")
    print(f"  framepipe {report['framepipe']['backend']}: {report['framepipe']['batch_fps']:.1f} images/s "
          f"(decode + edges + PNG write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
