#!/usr/bin/env python3
"""Compares the C++ implementation with the independent NumPy oracle.

For each case: writes a deterministic PNG with the oracle's own PNG writer,
computes all stages with scripts/oracle.py (float64), then runs
`framepipe validate --candidate <backend> --reference oracle:DIR`, which
compares every C++ stage (including signed Sobel components) against the
oracle with the normative per-stage tolerances. The oracle never calls C++.

Exit codes: 0 pass, 1 failure, 77 skipped (NumPy missing).
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
except ImportError:
    print("SKIP: NumPy not installed; oracle comparison not executed", file=sys.stderr)
    sys.exit(77)

import oracle  # noqa: E402
from generate_fixtures import pattern  # noqa: E402

# (pattern, in_w, in_h, out_w, out_h, radius, sigma, rgb, seed)
CASES = [
    ("constant", 1, 1, 1, 1, 3, 1.2, False, 0),
    ("random", 1, 7, 3, 5, 1, 0.5, True, 1),
    ("random", 7, 1, 5, 3, 3, 1.2, True, 2),
    ("random", 2, 2, 3, 3, 1, 1.0, False, 3),
    ("random", 3, 3, 3, 3, 3, 1.2, True, 4),
    ("checkerboard", 15, 17, 15, 17, 3, 1.2, True, 5),
    ("checkerboard", 16, 16, 16, 16, 0, 1.2, True, 6),
    ("stripes", 17, 19, 31, 33, 7, 3.0, True, 7),
    ("hramp", 31, 33, 17, 19, 15, 100.0, False, 8),
    ("random", 257, 129, 128, 64, 3, 1.2, True, 9),
    ("impulse", 64, 48, 1, 1, 3, 1.2, True, 10),
    ("vramp", 19, 23, 41, 7, 15, 0.1, True, 11),
    ("scene", 640, 480, 320, 240, 3, 1.2, True, 12),
    ("pure_rgb", 9, 3, 9, 3, 1, 1.2, True, 13),
    ("scene", 191, 107, 127, 71, 7, 2.0, False, 14),
    ("random", 40, 30, 64, 48, 1, 1.2, True, 15),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framepipe", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--candidates", default="cpu-ref,cpu-omp")
    a = ap.parse_args()
    os.makedirs(a.work_dir, exist_ok=True)
    doctor = json.loads(subprocess.run([a.framepipe, "doctor", "--json"], capture_output=True, text=True, check=True).stdout)
    usable = {b["backend"]: b["usable"] for b in doctor["backends"]}
    candidates = [c for c in a.candidates.split(",") if c]
    failures = 0
    ran = 0
    for i, (pat, w, h, wo, ho, r, s, rgb, seed) in enumerate(CASES):
        img = pattern(pat, w, h, seed, rgb=rgb)
        case = os.path.join(a.work_dir, f"case{i:02d}")
        os.makedirs(case, exist_ok=True)
        png = os.path.join(case, "input.png")
        oracle.write_png(png, img)
        for stage in ("resize", "blur", "edges"):
            odir = os.path.join(case, f"oracle_{stage}")
            os.makedirs(odir, exist_ok=True)
            for name, arr in oracle.run_pipeline(img, wo, ho, r, s, stage).items():
                oracle.write_f32(os.path.join(odir, name + ".f32"), arr)
            for cand in candidates:
                if not usable.get(cand, False):
                    print(f"SKIP case{i:02d} {stage} {cand}: backend not usable in this build", file=sys.stderr)
                    continue
                rep = os.path.join(case, f"report_{stage}_{cand}.json")
                cmd = [a.framepipe, "validate", "--input", png, "--candidate", cand, "--reference", "oracle:" + odir,
                       "--width", str(wo), "--height", str(ho), "--radius", str(r), "--sigma", repr(s),
                       "--output-stage", stage, "--report", rep, "--overwrite"]
                if cand == "cpu-omp":
                    cmd += ["--threads", "3"]
                p = subprocess.run(cmd, capture_output=True, text=True)
                ran += 1
                if p.returncode != 0:
                    failures += 1
                    print(f"FAIL case{i:02d} {pat} {w}x{h}->{wo}x{ho} r={r} s={s} seed={seed} stage={stage} "
                          f"cand={cand} rc={p.returncode}\n{p.stderr}", file=sys.stderr)
                    continue
                report = json.load(open(rep))
                kinds = {st["stage"]: st["status"] for st in report["stages"]}
                expected = ["gray", "resize"] + (["blur_horizontal", "blur"] if stage != "resize" else []) + \
                           (["sobel_x", "sobel_y", "edges"] if stage == "edges" else [])
                missing = [e for e in expected if kinds.get(e) != "passed"]
                if missing or report["validation_kind"] != "candidate_vs_independent_oracle":
                    failures += 1
                    print(f"FAIL case{i:02d}: stages not all passed: {kinds}", file=sys.stderr)
    print(f"oracle suite: {ran} comparisons, {failures} failures", file=sys.stderr)
    if ran == 0:
        print("SKIP: no candidate backend usable", file=sys.stderr)
        return 77
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
