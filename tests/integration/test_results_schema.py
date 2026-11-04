#!/usr/bin/env python3
"""Benchmark result-directory tests: schema validity, recomputation from raw
samples, honest handling of unavailable backends, and refusal to chart an
unmeasured report.

Usage: test_results_schema.py <framepipe> <source-dir> <work-dir>
"""

import json
import os
import shutil
import subprocess
import sys

FP, SRC, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
PY = sys.executable
failures = []


def check(c, m):
    if not c:
        failures.append(m)


shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
fix = os.path.join(SRC, "tests", "fixtures")
with open(os.path.join(WORK, "m.jsonl"), "w") as f:
    f.write(json.dumps({"frame_id": "checker", "path": os.path.join(fix, "checkerboard_rgb.png")}) + "\n")
    f.write(json.dumps({"frame_id": "random", "path": os.path.join(fix, "random_rgb_257x129.png")}) + "\n")

out = os.path.join(WORK, "run")
p = subprocess.run([FP, "benchmark", "--manifest", os.path.join(WORK, "m.jsonl"), "--backends",
                    "cpu-ref,cpu-omp,cuda,opencv-cpu", "--scopes", "host_processing,device_pipeline,application,startup,batch",
                    "--width", "64", "--height", "48", "--warmup", "2", "--iterations", "5", "--repetitions", "3",
                    "--threads", "2", "--output-dir", out], capture_output=True, text=True)
check(p.returncode in (0,), f"benchmark rc={p.returncode}: {p.stderr}")
check(p.stdout == "", "benchmark must not print to stdout")

v = subprocess.run([PY, os.path.join(SRC, "scripts", "validate_results.py"), out], capture_output=True, text=True)
check(v.returncode == 0, f"result dir invalid: {v.stderr}")

# Refuse to overwrite
p2 = subprocess.run([FP, "benchmark", "--manifest", os.path.join(WORK, "m.jsonl"), "--output-dir", out],
                    capture_output=True, text=True)
check(p2.returncode == 3, f"overwrite refusal rc={p2.returncode}")

# Tampering is detected by recomputation.
tam = os.path.join(WORK, "tampered")
shutil.copytree(out, tam)
s = json.load(open(os.path.join(tam, "summary.json")))
s["groups"][0]["pooled_within_runs"]["median_ms"] *= 0.5
json.dump(s, open(os.path.join(tam, "summary.json"), "w"))
v = subprocess.run([PY, os.path.join(SRC, "scripts", "validate_results.py"), tam], capture_output=True, text=True)
check(v.returncode == 1, "tampered summary must be rejected")

# NaN in samples rejected
nan = os.path.join(WORK, "nan")
shutil.copytree(out, nan)
with open(os.path.join(nan, "samples.jsonl"), "a") as f:
    f.write('{"duration_ms": NaN}\n')
v = subprocess.run([PY, os.path.join(SRC, "scripts", "validate_results.py"), nan], capture_output=True, text=True)
check(v.returncode == 1, "NaN sample must be rejected")

# Analysis script: recomputes from samples; refuses to chart not_run.
an = subprocess.run([PY, os.path.join(SRC, "scripts", "analyze_results.py"), out, "--no-plots"],
                    capture_output=True, text=True)
check(an.returncode == 0, f"analyze failed: {an.stderr}")
shell = os.path.join(WORK, "shell")
os.makedirs(shell)
shutil.copy(os.path.join(SRC, "benchmarks", "schema", "not_run_example.json"), os.path.join(shell, "summary.json"))
an = subprocess.run([PY, os.path.join(SRC, "scripts", "analyze_results.py"), shell], capture_output=True, text=True)
check(an.returncode == 2 and "not_run" in an.stderr, f"analyze must refuse not_run: rc={an.returncode} {an.stderr}")

doctor = json.loads(subprocess.run([FP, "doctor", "--json"], capture_output=True, text=True).stdout)
if not {b["backend"]: b["usable"] for b in doctor["backends"]}["cuda"]:
    summ = json.load(open(os.path.join(out, "summary.json")))
    cuda = [b for b in summ["backends"] if b["backend"] == "cuda"][0]
    check(cuda["status"] == "unavailable" and cuda["reason"], "cuda recorded unavailable with reason")
    check(all(r["host_processing_speedup_vs_scalar_cpu_ref"] is None for r in summ["ratios"]), "no GPU speedup")
    check(summ["status"] == "measured", f"CPU-only run status {summ['status']}")

shutil.rmtree(WORK, ignore_errors=True)
if failures:
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    sys.exit(1)
print("results schema checks passed", file=sys.stderr)
