#!/usr/bin/env python3
"""Generates the README showcase assets in docs/assets/.

Usage:
  make_showcase.py hero    --framepipe build/cpu/bin/framepipe [--input PHOTO]
  make_showcase.py figures [--evidence results/evidence/2026-11-05_kaggle-t4_gpu_matrix]
  make_showcase.py all     --framepipe build/cpu/bin/framepipe

hero     Runs the real `framepipe process` CLI (cpu-ref) on one input at each
         output stage and composes input | resize | blur | edges into
         docs/assets/hero.png. Without --input, a procedural 1920x1080 scene is
         rendered (no third-party image, so no licensing questions). Pass your
         own photo with --input to use it instead.
figures  Draws charts from MEASURED data only: the committed summary.json files
         of a benchmark evidence directory. Nothing is typed in by hand.

Requires numpy and matplotlib (requirements.txt).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")
DEFAULT_EVIDENCE = os.path.join(ROOT, "results", "evidence", "2026-11-05_kaggle-t4_gpu_matrix")

# Consistent backend colours across every figure.
COLORS = {
    "cpu-ref": "#8c8c8c",
    "cpu-omp": "#4c78a8",
    "opencv-cpu": "#f58518",
    "cuda": "#54a24b",
}
LABELS = {
    "cpu-ref": "cpu-ref (scalar)",
    "cpu-omp": "cpu-omp (4 threads)",
    "opencv-cpu": "OpenCV (1 thread)",
    "cuda": "CUDA (incl. transfers)",
}


# --------------------------------------------------------------------------- hero


def render_scene(path, w=1920, h=1080):
    """A procedural scene with strong, varied edges: shapes, text, a grid, gradients."""
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    # Sky-to-ground gradient background.
    y = np.linspace(0, 1, h)[:, None]
    top, bottom = np.array([0.18, 0.32, 0.55]), np.array([0.93, 0.72, 0.45])
    bg = top * (1 - y[..., None]) + bottom * y[..., None]
    ax.imshow(np.broadcast_to(bg, (h, w, 3)), extent=(0, w, h, 0))

    # Sun with concentric rings.
    for r, a in [(170, 0.25), (130, 0.5), (95, 1.0)]:
        ax.add_patch(Circle((1500, 260), r, color=(1.0, 0.85, 0.35), alpha=a, lw=0))

    # City skyline.
    rng = np.random.default_rng(7)
    x = 0
    while x < w:
        bw = int(rng.integers(70, 160))
        bh = int(rng.integers(220, 560))
        shade = rng.uniform(0.12, 0.3)
        ax.add_patch(Rectangle((x, h - 220 - bh), bw - 6, bh, color=(shade, shade, shade + 0.05), lw=0))
        for wy in range(h - 200 - bh, h - 240, 38):
            for wx in range(x + 12, x + bw - 24, 26):
                if rng.random() < 0.55:
                    ax.add_patch(Rectangle((wx, wy), 12, 18, color=(1.0, 0.9, 0.55), lw=0))
        x += bw

    # Ground plane with a perspective grid.
    ax.add_patch(Rectangle((0, h - 220), w, 220, color=(0.22, 0.2, 0.24), lw=0))
    vx = w / 2
    for i in range(-12, 13):
        ax.plot([vx, vx + i * 220], [h - 220, h], color=(0.85, 0.85, 0.9), lw=2)
    for t in [0.08, 0.2, 0.38, 0.62, 1.0]:
        yy = h - 220 + 220 * t
        ax.plot([0, w], [yy, yy], color=(0.85, 0.85, 0.9), lw=2)

    # Foreground objects: a triangle sign and a checkerboard target.
    ax.add_patch(Polygon([(260, 700), (420, 430), (580, 700)], closed=True, color=(0.95, 0.3, 0.25), lw=0))
    ax.add_patch(Polygon([(310, 670), (420, 480), (530, 670)], closed=True, color=(1, 1, 1), lw=0))
    s = 34
    for i in range(6):
        for j in range(6):
            c = 0.05 if (i + j) % 2 else 0.95
            ax.add_patch(Rectangle((1600 + i * s, 560 + j * s), s, s, color=(c, c, c), lw=0))

    ax.text(w / 2, 150, "FramePipe", ha="center", va="center", fontsize=110,
            fontweight="bold", color=(1, 1, 1), family="DejaVu Sans")
    # FramePipe rejects alpha (by design), so drop matplotlib's alpha channel.
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image  # matplotlib's own PNG backend dependency
    Image.fromarray(rgb, "RGB").save(path)


def run_stage(fp, inp, out, stage):
    cmd = [fp, "process", "--input", inp, "--output", out, "--backend", "cpu-ref",
           "--config", os.path.join(ROOT, "configs", "edges_720p.json"),
           "--output-stage", stage, "--overwrite"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def load_rgb(path):
    """PNG/JPEG -> float RGB in [0,1] (gray inputs are broadcast)."""
    img = plt.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3]


def cmd_hero(a):
    if not a.framepipe or not os.access(a.framepipe, os.X_OK):
        sys.exit("make_showcase: --framepipe must point to a built framepipe binary")
    os.makedirs(ASSETS, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        inp = a.input
        if not inp:
            inp = os.path.join(tmp, "scene.png")
            render_scene(inp)
        outs = {}
        for stage in ("resize", "blur", "edges"):
            outs[stage] = os.path.join(tmp, f"{stage}.png")
            run_stage(a.framepipe, inp, outs[stage], stage)
        edges = load_rgb(outs["edges"])
        # The pipeline's edge PNG is normalized to the theoretical maximum, so
        # real edges are dim at thumbnail size. Stretch for DISPLAY only, and say so.
        hi = float(np.percentile(edges, 99.5)) or 1.0
        edges_disp = np.clip(edges / hi, 0, 1) ** 0.7
        panels = [("input (RGB)", load_rgb(inp)),
                  ("grayscale + bilinear resize", load_rgb(outs["resize"])),
                  ("Gaussian blur (r=3, \u03c3=1.2)", load_rgb(outs["blur"])),
                  ("Sobel edge magnitude (display-stretched)", edges_disp)]
        fig, axes = plt.subplots(1, 4, figsize=(20, 3.6), dpi=110)
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, interpolation="lanczos")
            ax.set_title(title, fontsize=13)
            ax.axis("off")
        for i in range(3):
            fig.text(0.255 + i * 0.245, 0.47, "\u2192", fontsize=26, ha="center", va="center")
        fig.suptitle("framepipe process --config configs/edges_720p.json   (1920\u00d71080 \u2192 1280\u00d7720)",
                     fontsize=11, family="monospace", y=0.04)
        fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.5)
        p = os.path.join(ASSETS, "hero.png")
        fig.savefig(p, facecolor="white")
        plt.close(fig)
        print(f"make_showcase: wrote {p}", file=sys.stderr)
        # Full-resolution, unmodified pipeline output for anyone who wants to look closely.
        p2 = os.path.join(ASSETS, "edges_1280x720.png")
        subprocess.run(["cp", outs["edges"], p2], check=True)
        print(f"make_showcase: wrote {p2}", file=sys.stderr)


# ----------------------------------------------------------------------- figures


def load_medians(evidence, scenario):
    """{(backend, metric): median_ms} for the first non-batch frame of a scenario."""
    d = os.path.join(evidence, scenario)
    summary = json.load(open(os.path.join(d, "summary.json")))
    if summary.get("status") != "measured":
        raise SystemExit(f"make_showcase: {d} status is {summary.get('status')}; refusing to chart")
    out = {}
    for g in summary["groups"]:
        if g["frame_id"] == "__batch__":
            continue
        out.setdefault((g["backend"], g["metric"]), g["pooled_within_runs"]["median_ms"])
    return out


def hardware_note(evidence):
    env = json.load(open(os.path.join(evidence, "hd", "environment.json")))
    gpus = ", ".join(sorted({g["name"] for g in env.get("cuda", {}).get("devices", [])}))
    return f"{gpus} + {env['host']['cpu_model']} ({env['host']['logical_cpus']} vCPU)"


def footer(fig, evidence, extra=""):
    rel = os.path.relpath(evidence, ROOT)
    c = json.load(open(os.path.join(evidence, "hd", "summary.json")))["counts"]
    n = c["iterations"] * c["repetitions"]
    fig.text(0.01, 0.01, f"Measured: {hardware_note(evidence)}. Median of {n} samples per bar. "
             f"Source: {rel}/ {extra}", fontsize=7, color="#555")


def fig_core_latency(evidence):
    """Core processing time per backend, per scenario (log scale)."""
    scen = [("tiny", "64\u00b2"), ("small", "640\u00d7480"), ("hd", "1080p\u2192720p"),
            ("uhd", "4K\u21921080p"), ("upscale", "480p\u2192960p")]
    metric = {"cpu-ref": "cpu_core_wall_ms", "cpu-omp": "cpu_core_wall_ms",
              "opencv-cpu": "cpu_core_wall_ms", "cuda": "gpu_transfer_inclusive_wall_ms"}
    backends = list(metric)
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=120)
    wdt = 0.2
    for bi, b in enumerate(backends):
        vals = [load_medians(evidence, s)[(b, metric[b])] for s, _ in scen]
        xs = [i + (bi - 1.5) * wdt for i in range(len(scen))]
        bars = ax.bar(xs, vals, wdt, color=COLORS[b], label=LABELS[b])
        for x, v in zip(xs, vals):
            ax.text(x, v * 1.08, f"{v:.2g}" if v < 10 else f"{v:.0f}", ha="center", fontsize=7)
        del bars
    ax.set_yscale("log")
    ax.set_xticks(range(len(scen)))
    ax.set_xticklabels([lab for _, lab in scen])
    ax.set_ylabel("core processing time (ms, log scale)")
    ax.set_title("Grayscale \u2192 resize \u2192 blur \u2192 Sobel: core time per frame (lower is better)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    footer(fig, evidence)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_gpu_breakdown(evidence):
    """Where CUDA time goes: kernels vs host<->device transfers."""
    scen = [("tiny", "64\u00b2"), ("small", "640\u00d7480"), ("hd", "1080p\u2192720p"),
            ("same", "1080p\u21921080p"), ("upscale", "480p\u2192960p"), ("uhd", "4K\u21921080p")]
    kern, xfer = [], []
    for s, _ in scen:
        m = load_medians(evidence, s)
        k = m[("cuda", "gpu_device_pipeline_ms")]
        t = m[("cuda", "gpu_transfer_inclusive_wall_ms")]
        kern.append(k)
        xfer.append(max(t - k, 0.0))
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=120)
    y = np.arange(len(scen))
    ax.barh(y, kern, color="#54a24b", label="kernels (device-resident time)")
    ax.barh(y, xfer, left=kern, color="#bab0ac", label="upload + download + sync (inclusive \u2212 resident)")
    for i, (k, x) in enumerate(zip(kern, xfer)):
        ax.text(k + x + 0.08, i, f"{k + x:.2f} ms  ({100 * k / (k + x):.0f}% kernels)", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in scen])
    ax.invert_yaxis()
    ax.set_xlabel("CUDA time per frame (ms)")
    ax.set_xlim(0, max(k + x for k, x in zip(kern, xfer)) * 1.35)
    ax.set_title("Where the GPU time goes: transfers, not kernels")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    footer(fig, evidence)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_end_to_end(evidence):
    """Core speedup vs application speedup on HD: the I/O bound."""
    m = load_medians(evidence, "hd")
    rows = [("cpu-ref", m[("cpu-ref", "cpu_core_wall_ms")], m[("cpu-ref", "application_wall_ms")]),
            ("cpu-omp", m[("cpu-omp", "cpu_core_wall_ms")], m[("cpu-omp", "application_wall_ms")]),
            ("cuda", m[("cuda", "gpu_transfer_inclusive_wall_ms")], m[("cuda", "application_wall_ms")])]
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=120)
    y = np.arange(len(rows))
    for i, (b, core, app) in enumerate(rows):
        ax.barh(i, core, color=COLORS[b])
        ax.barh(i, app - core, left=core, color=COLORS[b], alpha=0.3, hatch="//", edgecolor="white")
        ax.text(app + 1, i, f"{app:.0f} ms total, {core:.1f} ms processing", va="center", fontsize=8)
    ref_core, ref_app = rows[0][1], rows[0][2]
    cu_core, cu_app = rows[2][1], rows[2][2]
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[b] for b, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, ref_app * 1.45)
    ax.set_xlabel("HD frame, end to end (ms): solid = processing, hatched = decode, quantize, PNG encode, file I/O")
    ax.set_title(f"CUDA processing is {ref_core / cu_core:.1f}\u00d7 faster, "
                 f"but the whole frame is only {ref_app / cu_app:.2f}\u00d7 faster")
    ax.spines[["top", "right"]].set_visible(False)
    footer(fig, evidence)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def cmd_figures(a):
    os.makedirs(ASSETS, exist_ok=True)
    for name, fn in [("core_latency", fig_core_latency), ("gpu_breakdown", fig_gpu_breakdown),
                     ("end_to_end", fig_end_to_end)]:
        fig = fn(a.evidence)
        p = os.path.join(ASSETS, f"{name}.png")
        fig.savefig(p, facecolor="white")
        plt.close(fig)
        print(f"make_showcase: wrote {p}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["hero", "figures", "all"])
    ap.add_argument("--framepipe", help="path to a built framepipe binary (hero)")
    ap.add_argument("--input", help="your own photo for the hero image (default: procedural scene)")
    ap.add_argument("--evidence", default=DEFAULT_EVIDENCE, help="benchmark evidence directory (figures)")
    a = ap.parse_args()
    if a.what in ("hero", "all"):
        cmd_hero(a)
    if a.what in ("figures", "all"):
        cmd_figures(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
