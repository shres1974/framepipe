#!/usr/bin/env python3
"""Re-derives statistics from raw samples and draws figures from MEASURED data only.

Usage:
  analyze_results.py RESULT_DIR [RESULT_DIR ...] [--out DIR] [--no-plots]

For each result directory, recomputes per-(backend, metric, frame) pooled
median / p95 (nearest rank) / count directly from samples.jsonl, writes
recomputed_summary.csv, and (unless --no-plots) a latency chart per metric.
Refuses (exit 2) to chart any directory whose summary status is "not_run" or
that has no successful samples. Every figure states units, workload,
hardware, configuration hash, sample counts, and its source directory.
"""

import argparse
import gzip
import csv
import json
import math
import os
import sys
from collections import defaultdict


def open_samples(d):
    """samples.jsonl, or samples.jsonl.gz in archived evidence directories."""
    p = os.path.join(d, "samples.jsonl")
    if os.path.exists(p):
        return open(p)
    return gzip.open(p + ".gz", "rt")


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def p95(v):
    v = sorted(v)
    return v[math.ceil(0.95 * len(v)) - 1]


class RefuseError(Exception):
    pass


def load_dir(d):
    summary = json.load(open(os.path.join(d, "summary.json")))
    if summary.get("status") == "not_run":
        raise RefuseError(f"{d}: status is not_run; refusing to create measured charts")
    samples = []
    with open_samples(d) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    ok = [s for s in samples if s["success"]]
    if not ok:
        raise RefuseError(f"{d}: no successful samples; refusing to chart")
    env = json.load(open(os.path.join(d, "environment.json")))
    return summary, ok, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None, help="output directory (default: first RESULT_DIR)")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    out = a.out or a.dirs[0]
    os.makedirs(out, exist_ok=True)
    rows = []
    hw = set()
    hashes = set()
    try:
        for d in a.dirs:
            summary, ok, env = load_dir(d)
            cpu = env["host"].get("cpu_model") or "unknown CPU"
            gpus = [g["name"] for g in env["cuda"].get("devices", [])]
            hw.add(cpu + (" + " + ", ".join(gpus) if gpus else ""))
            hashes.add(summary["math_config_hash"][:12])
            groups = defaultdict(list)
            meta = {}
            for s in ok:
                k = (s["backend"], s["implementation"], s["metric"], s["frame_id"])
                groups[k].append(s["duration_ms"])
                meta[k] = (s["input_width"], s["input_height"], s["output_width"], s["output_height"], s["workload"])
            for k, v in sorted(groups.items()):
                iw, ih, ow, oh, wl = meta[k]
                rows.append(dict(source=d, backend=k[0], implementation=k[1], metric=k[2], frame_id=k[3],
                                 input=f"{iw}x{ih}", output=f"{ow}x{oh}", workload=wl, count=len(v),
                                 median_ms=median(v), p95_ms=p95(v), min_ms=min(v), max_ms=max(v)))
    except RefuseError as e:
        print(f"analyze_results: {e}", file=sys.stderr)
        return 2
    with open(os.path.join(out, "recomputed_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"analyze_results: {len(rows)} groups -> {os.path.join(out, 'recomputed_summary.csv')}", file=sys.stderr)
    if a.no_plots:
        return 0
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("analyze_results: matplotlib not installed; skipping plots", file=sys.stderr)
        return 0
    by_metric = defaultdict(list)
    for r in rows:
        if r["metric"] != "batch_wall_ms":
            by_metric[r["metric"]].append(r)
    for metric, rs in by_metric.items():
        labels = sorted({(r["input"], r["output"]) for r in rs}, key=lambda t: [int(x) for x in t[0].split("x")])
        backends = sorted({(r["backend"], r["implementation"]) for r in rs})
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels) + 3), 4.5))
        width = 0.8 / max(1, len(backends))
        for bi, (b, impl) in enumerate(backends):
            meds, p95s, ns = [], [], []
            for lab in labels:
                m = [r for r in rs if (r["input"], r["output"]) == lab and r["backend"] == b and r["implementation"] == impl]
                meds.append(median([x["median_ms"] for x in m]) if m else float("nan"))
                p95s.append(max(x["p95_ms"] for x in m) if m else float("nan"))
                ns.append(sum(x["count"] for x in m))
            xs = [i + bi * width for i in range(len(labels))]
            ax.bar(xs, meds, width, label=f"{b} [{impl}]")
            ax.scatter(xs, p95s, marker="_", color="black", s=60, zorder=3)
        ax.set_xticks([i + width * (len(backends) - 1) / 2 for i in range(len(labels))])
        ax.set_xticklabels([f"{i}\n-> {o}" for i, o in labels], fontsize=8)
        ax.set_ylabel(f"{metric} (milliseconds, median; tick = p95)")
        ax.set_yscale("log")
        n_total = sum(r["count"] for r in rs)
        ax.set_title(f"{metric}: measured on {'; '.join(sorted(hw))}\n"
                     f"config hash {', '.join(sorted(hashes))}, n={n_total} samples, workload "
                     f"{', '.join(sorted({r['workload'] for r in rs}))}", fontsize=8)
        ax.legend(fontsize=7)
        fig.text(0.01, 0.01, "source: " + ", ".join(a.dirs), fontsize=6)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        p = os.path.join(out, f"{metric}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"analyze_results: wrote {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
