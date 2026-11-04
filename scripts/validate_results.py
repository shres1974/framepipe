#!/usr/bin/env python3
"""Validates a FramePipe benchmark result directory against
benchmarks/schema/results_v1.schema.json and checks cross-file invariants:

  * all numbers finite (no NaN/Infinity tokens)
  * every sample's metric matches its backend (no device_pipeline for CPU)
  * summary statistics recompute from raw samples (median, p95 nearest rank, count)
  * unavailable backends have zero samples
  * success-only aggregates exclude failed samples

Stdlib only. Usage: validate_results.py RESULT_DIR [--schema PATH]
Exit 0 valid, 1 invalid.
"""

import argparse
import gzip
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCHEMA = os.path.join(HERE, "..", "benchmarks", "schema", "results_v1.schema.json")
errors = []


def _reject_constant(tok):
    raise ValueError(f"non-finite JSON token {tok}")


def load(path):
    with open(path) as f:
        return json.load(f, parse_constant=_reject_constant)


TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def type_ok(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, TYPES[t])


def check(v, s, where):
    t = s.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(type_ok(v, x) for x in ts):
            errors.append(f"{where}: expected {t}, got {type(v).__name__}")
            return
    if "enum" in s and v not in s["enum"]:
        errors.append(f"{where}: {v!r} not in {s['enum']}")
    if "minimum" in s and isinstance(v, (int, float)) and v < s["minimum"]:
        errors.append(f"{where}: {v} < minimum {s['minimum']}")
    if isinstance(v, float) and not math.isfinite(v):
        errors.append(f"{where}: non-finite number")
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                errors.append(f"{where}: missing required '{k}'")
        for k, sub in s.get("properties", {}).items():
            if k in v:
                check(v[k], sub, f"{where}.{k}")
    if isinstance(v, list) and "items" in s:
        for i, x in enumerate(v):
            check(x, s["items"], f"{where}[{i}]")


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


def validate_dir(d, schema):
    for name in ("run.json", "environment.json", "validation.json", "summary.json", "resolved_config.json"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            errors.append(f"missing {name}")
            continue
        try:
            doc = load(p)
        except ValueError as e:
            errors.append(f"{name}: {e}")
            continue
        if name in schema:
            check(doc, schema[name], name)
    for name in ("inputs.jsonl", "samples.jsonl", "summary.csv", "analysis.md"):
        if not (os.path.exists(os.path.join(d, name)) or os.path.exists(os.path.join(d, name + ".gz"))):
            errors.append(f"missing {name}")
    if errors:
        return
    samples = []
    with open_samples(d) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            s = json.loads(line, parse_constant=_reject_constant)
            check(s, schema["sample"], f"samples.jsonl:{i + 1}")
            samples.append(s)
    summary = load(os.path.join(d, "summary.json"))
    gpu_metrics = {"gpu_transfer_inclusive_wall_ms", "gpu_device_pipeline_ms"}
    for s in samples:
        if s["backend"] != "cuda" and s["metric"] in gpu_metrics:
            errors.append(f"sample metric {s['metric']} on non-GPU backend {s['backend']}")
        if s["backend"] == "cuda" and s["metric"] == "cpu_core_wall_ms":
            errors.append("cuda sample labeled cpu_core_wall_ms")
    status = {b["backend"]: b["status"] for b in summary["backends"]}
    for b, st in status.items():
        if st == "unavailable" and any(s["backend"] == b for s in samples):
            errors.append(f"unavailable backend {b} has samples")
    # Recompute pooled statistics from raw successful samples.
    for g in summary["groups"]:
        vals = [s["duration_ms"] for s in samples if s["success"] and s["backend"] == g["backend"]
                and s["implementation"] == g["implementation"] and s["metric"] == g["metric"]
                and s["frame_id"] == g["frame_id"]]
        pw = g["pooled_within_runs"]
        if len(vals) != pw["count"]:
            errors.append(f"group {g['backend']}/{g['metric']}/{g['frame_id']}: count {pw['count']} != {len(vals)}")
            continue
        for k, f in (("median_ms", median), ("p95_ms", p95), ("min_ms", min), ("max_ms", max)):
            if not math.isclose(pw[k], f(vals), rel_tol=1e-9, abs_tol=1e-12):
                errors.append(f"group {g['backend']}/{g['metric']}/{g['frame_id']}: {k} {pw[k]} != {f(vals)}")
    if summary["status"] != "not_run" and not samples:
        errors.append("summary claims measurements but samples.jsonl is empty")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    a = ap.parse_args()
    schema = load(a.schema)
    validate_dir(a.result_dir, schema)
    if errors:
        for e in errors:
            print("INVALID:", e, file=sys.stderr)
        return 1
    print(f"valid: {a.result_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
