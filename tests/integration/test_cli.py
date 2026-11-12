#!/usr/bin/env python3
"""CLI integration tests: commands, exit codes, stdout hygiene, overwrite,
color order, unsupported formats, configuration precedence, batch.

Usage: test_cli.py <framepipe-binary> <source-dir>
Uses only the standard library (no NumPy) so it always runs.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

FP = sys.argv[1]
SRC = sys.argv[2]
FIX = os.path.join(SRC, "tests", "fixtures")
failures = []


def run(*args, ok=None):
    p = subprocess.run([FP, *args], capture_output=True, text=True)
    if ok is not None and p.returncode != ok:
        failures.append(f"{' '.join(args)}: expected rc {ok}, got {p.returncode}\nstderr: {p.stderr}")
    return p


def check(cond, msg):
    if not cond:
        failures.append(msg)


def png(path, w, h, ctype, depth, pixels: bytes):
    ch = {0: 1, 2: 3, 6: 4, 4: 2}[ctype]
    bpp = ch * depth // 8
    raw = b"".join(b"\x00" + pixels[y * w * bpp:(y + 1) * w * bpp] for y in range(h))

    def chunk(t, b):
        return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)

    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    open(path, "wb").write(data)


def read_png_gray(path):
    data = open(path, "rb").read()
    pos, idat = 8, b""
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        t = data[pos + 4:pos + 8]
        if t == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", data[pos + 8:pos + 18])
        if t == b"IDAT":
            idat += data[pos + 8:pos + 8 + n]
        pos += 12 + n
    assert depth == 8 and ctype == 0, (depth, ctype)
    raw = zlib.decompress(idat)
    rows, prev = [], bytearray(w)
    for y in range(h):
        f = raw[y * (w + 1)]
        line = bytearray(raw[y * (w + 1) + 1:(y + 1) * (w + 1)])
        for i in range(w):
            a = line[i - 1] if i else 0
            b = prev[i]
            c = prev[i - 1] if i else 0
            if f == 1: line[i] = (line[i] + a) & 255
            elif f == 2: line[i] = (line[i] + b) & 255
            elif f == 3: line[i] = (line[i] + ((a + b) >> 1)) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else (b if pb <= pc else c))) & 255
        rows.append(bytes(line))
        prev = line
    return w, h, rows


T = tempfile.mkdtemp(prefix="framepipe_cli_")
try:
    # ------------------------------------------------------------- help / top level
    p = run("--help", ok=0)
    check("doctor" in p.stdout and "benchmark" in p.stdout, "top help lists commands")
    run(ok=2)
    run("frobnicate", ok=2)
    for cmd in ("doctor", "process", "batch", "validate", "benchmark"):
        p = run(cmd, "--help", ok=0)
        check("Usage:" in p.stdout, f"{cmd} --help prints usage")

    # ------------------------------------------------------------- doctor
    p = run("doctor", "--json", ok=0)
    d = json.loads(p.stdout)  # stdout must be ONLY valid JSON
    names = {b["backend"]: b for b in d["backends"]}
    check(names["cpu-ref"]["usable"], "cpu-ref usable")
    check("cuda" in names and "reason" in names["cuda"], "doctor reports cuda status with reason")
    check("hip" not in names, "doctor must not advertise hip")
    check(d["auto_selection"]["selected"] in ("cpu-ref", "cuda"), "auto selection reported")
    p = run("doctor", ok=0)
    check(p.stdout == "", "human doctor output goes to stderr")
    cuda_usable = names["cuda"]["usable"]

    # ------------------------------------------------------------- process
    inp = os.path.join(FIX, "checkerboard_rgb.png")
    out = os.path.join(T, "nested", "dir", "edges.png")
    rep = os.path.join(T, "nested", "report.json")
    run("process", "--input", inp, "--output", out, "--backend", "cpu-ref", "--width", "32", "--height", "24",
        "--report", rep, ok=0)
    w, h, _ = read_png_gray(out)
    check((w, h) == (32, 24), "process output size")
    r = json.load(open(rep))
    check(r["selected_backend"] == "cpu-ref" and r["resolved_config"]["resize"]["width"] == 32, "report config")
    check(r["input"]["orientation_policy"].startswith("stored"), "orientation policy recorded")
    # overwrite protection
    run("process", "--input", inp, "--output", out, "--backend", "cpu-ref", ok=3)
    run("process", "--input", inp, "--output", out, "--backend", "cpu-ref", "--width", "32", "--height", "24",
        "--report", rep, "--overwrite", ok=0)
    # same input/output
    tmp_in = os.path.join(T, "same.png")
    shutil.copy(inp, tmp_in)
    run("process", "--input", tmp_in, "--output", tmp_in, "--overwrite", "--backend", "cpu-ref", ok=2)
    run("process", "--input", tmp_in, "--output", os.path.join(T, ".", "same.png"), "--overwrite",
        "--backend", "cpu-ref", ok=2)
    # report to stdout is pure JSON
    p = run("process", "--input", inp, "--output", os.path.join(T, "o2.png"), "--backend", "cpu-ref",
            "--width", "8", "--height", "8", "--report", "-", ok=0)
    json.loads(p.stdout)

    # ------------------------------------------------------------- color order through CLI
    # 3x1 RGB: red, green, blue. resize stage, identity size, so outputs are the grayscale values.
    rgb = os.path.join(T, "rgb.png")
    png(rgb, 3, 1, 2, 8, bytes([255, 0, 0, 0, 255, 0, 0, 0, 255]))
    o = os.path.join(T, "rgb_out.png")
    run("process", "--input", rgb, "--output", o, "--backend", "cpu-ref", "--output-stage", "resize",
        "--width", "3", "--height", "1", ok=0)
    _, _, rows = read_png_gray(o)
    check(list(rows[0]) == [77, 149, 29], f"RGB order through CLI: got {list(rows[0])}")
    # gray 1x1 128 -> 128 after resize/blur
    o = os.path.join(T, "g.png")
    run("process", "--input", os.path.join(FIX, "gray_1x1_128.png"), "--output", o, "--backend", "cpu-ref",
        "--output-stage", "blur", "--width", "5", "--height", "4", ok=0)
    _, _, rows = read_png_gray(o)
    check(all(v == 128 for row in rows for v in row), "gray 128 preserved through blur + quantization")

    # ------------------------------------------------------------- unsupported / corrupt inputs
    rgba = os.path.join(T, "rgba.png")
    png(rgba, 2, 2, 6, 8, bytes(range(16)))
    p = run("process", "--input", rgba, "--output", os.path.join(T, "x.png"), "--backend", "cpu-ref", ok=3)
    check("alpha" in p.stderr, "alpha rejection message")
    g16 = os.path.join(T, "g16.png")
    png(g16, 2, 2, 0, 16, bytes(8))
    p = run("process", "--input", g16, "--output", os.path.join(T, "x.png"), "--backend", "cpu-ref", ok=3)
    check("16-bit" in p.stderr, "16-bit rejection message")
    bad = os.path.join(T, "bad.png")
    open(bad, "wb").write(b"\x89PNG\r\n\x1a\nnope")
    run("process", "--input", bad, "--output", os.path.join(T, "x.png"), "--backend", "cpu-ref", ok=3)
    run("process", "--input", os.path.join(T, "missing.png"), "--output", os.path.join(T, "x.png"), ok=3)
    # output failure: parent is a file
    run("process", "--input", inp, "--output", os.path.join(inp, "sub", "x.png"), "--backend", "cpu-ref", ok=3)
    # EXIF orientation ignored: stored 4x2 -> identity resize keeps 4x2
    o = os.path.join(T, "exif.png")
    run("process", "--input", os.path.join(FIX, "exif_orientation6.jpg"), "--output", o, "--backend", "cpu-ref",
        "--output-stage", "resize", "--width", "4", "--height", "2", "--report", os.path.join(T, "exif.json"), ok=0)
    check(json.load(open(os.path.join(T, "exif.json")))["input"]["width"] == 4, "EXIF orientation ignored")

    # ------------------------------------------------------------- checked-in configs vs schema and loader
    schema = json.load(open(os.path.join(SRC, "configs", "pipeline_config.schema.json")))

    def schema_ok(doc, s, where="$"):
        if s.get("type") == "object":
            if not isinstance(doc, dict):
                return f"{where} not object"
            for k in s.get("required", []):
                if k not in doc:
                    return f"{where}.{k} missing"
            props = s.get("properties", {})
            for k, v in doc.items():
                if k not in props:
                    return f"{where}.{k} not allowed"
                e = schema_ok(v, props[k], f"{where}.{k}")
                if e:
                    return e
        if "enum" in s and doc not in s["enum"]:
            return f"{where}={doc!r} not in enum"
        if "minimum" in s and doc < s["minimum"]:
            return f"{where} below minimum"
        if "maximum" in s and doc > s["maximum"]:
            return f"{where} above maximum"
        return None

    for name in sorted(os.listdir(os.path.join(SRC, "configs"))):
        if not name.endswith(".json") or name.endswith(".schema.json"):
            continue
        path = os.path.join(SRC, "configs", name)
        err = schema_ok(json.load(open(path)), schema)
        check(err is None, f"{name} violates schema: {err}")
        rpt = os.path.join(T, f"cfg_{name}.json")
        p = run("process", "--input", os.path.join(FIX, "gray_1x1_128.png"), "--output",
                os.path.join(T, f"cfg_{name}.png"), "--config", path, "--backend", "cpu-ref", "--report", rpt)
        check(p.returncode == 0, f"{name} rejected by loader: {p.stderr}")
    # The schema's backend enum must match what the loader accepts.
    check(schema["properties"]["backend"]["enum"] == ["auto", "cpu-ref", "cpu-omp", "cuda"], "schema backend enum")

    # ------------------------------------------------------------- configuration errors and precedence
    cfg = os.path.join(T, "c.json")
    json.dump({"schema_version": 1, "resize": {"width": 20, "height": 10}, "gaussian": {"radius": 2}}, open(cfg, "w"))
    run("process", "--input", inp, "--output", os.path.join(T, "p.png"), "--config", cfg, "--height", "6",
        "--backend", "cpu-ref", "--report", os.path.join(T, "p.json"), ok=0)
    pr = json.load(open(os.path.join(T, "p.json")))["resolved_config"]
    check(pr["resize"] == {"width": 20, "height": 6, "mode": "bilinear_half_pixel", "aspect_policy": "stretch"},
          f"precedence defaults<file<cli: {pr['resize']}")
    check(pr["gaussian"]["radius"] == 2, "file value kept")
    json.dump({"schema_version": 1, "gausian": {}}, open(cfg, "w"))
    p = run("process", "--input", inp, "--output", os.path.join(T, "q.png"), "--config", cfg, ok=2)
    check("gausian" in p.stderr, "unknown field named")
    run("process", "--input", inp, "--output", os.path.join(T, "q.png"), "--sigma", "nan", ok=2)
    run("process", "--input", inp, "--output", os.path.join(T, "q.png"), "--radius", "99", ok=2)
    run("process", "--input", inp, "--output", os.path.join(T, "q.png"), "--backend", "hip", ok=2)
    run("process", "--input", inp, "--output", os.path.join(T, "q.png"), "--bogus", ok=2)
    run("process", "--input", inp, "--output", os.path.join(T, "q.png"), "--width", "abc", ok=2)
    run("process", "--input", os.path.join(FIX, "checkerboard_rgb.png"), ok=2)  # missing --output

    # ------------------------------------------------------------- backend selection
    if not cuda_usable:
        p = run("process", "--input", inp, "--output", os.path.join(T, "cuda.png"), "--backend", "cuda", ok=4)
        check("cuda" in p.stderr, "explicit cuda failure names backend")
        check(not os.path.exists(os.path.join(T, "cuda.png")), "no output written on backend failure")
        run("validate", "--input", inp, "--candidate", "cuda", ok=4)
    run("process", "--input", inp, "--output", os.path.join(T, "auto.png"), "--backend", "auto",
        "--report", os.path.join(T, "auto.json"), "--width", "16", "--height", "16", ok=0)
    a = json.load(open(os.path.join(T, "auto.json")))
    check(a["selection_reason"].startswith("auto"), "auto reports reason")

    # ------------------------------------------------------------- validate
    vr = os.path.join(T, "v.json")
    run("validate", "--input", inp, "--report", vr, "--width", "40", "--height", "30", ok=0)
    v = json.load(open(vr))
    check(v["validation_kind"] == "interface_smoke_check_self_comparison", "self comparison labeled")
    check(all(s["status"] == "passed" for s in v["stages"]), "self comparison passes")
    check(any(s["stage"] == "sobel_x" for s in v["stages"]), "signed sobel components compared")
    for s in v["stages"]:
        if s["exact_match"]:
            check(s["psnr_db"] is None, "psnr null on exact match")
    if names["cpu-omp"]["usable"]:
        run("validate", "--input", inp, "--candidate", "cpu-omp", "--threads", "4", "--report", vr, "--overwrite",
            "--width", "40", "--height", "30", ok=0)
        check(json.load(open(vr))["validation_kind"] == "candidate_vs_cpu_ref", "omp vs ref kind")
    # A deliberately wrong oracle must produce exit 6.
    od = os.path.join(T, "wrong_oracle")
    os.makedirs(od)
    with open(os.path.join(od, "resize.f32"), "wb") as f:
        f.write(b"FPF32 4 3\n" + struct.pack("<12f", *([0.25] * 12)))
    p = run("validate", "--input", inp, "--reference", "oracle:" + od, "--output-stage", "resize",
            "--width", "4", "--height", "3", ok=6)
    check("FAILED" in p.stderr, "validate failure reported")

    # ------------------------------------------------------------- batch
    bd = os.path.join(T, "batch")
    os.makedirs(bd)
    for i, n in enumerate(["frame_9", "frame_10", "frame_2"]):
        shutil.copy(os.path.join(FIX, ["checkerboard_rgb.png", "vramp_rgb.png", "impulse_rgb.png"][i]),
                    os.path.join(bd, n + ".png"))
    with open(os.path.join(bd, "m.jsonl"), "w") as f:
        for n in ["frame_9", "frame_10", "frame_2"]:
            f.write(json.dumps({"frame_id": n, "path": n + ".png"}) + "\n")
    outd = os.path.join(T, "batch_out")
    run("batch", "--manifest", os.path.join(bd, "m.jsonl"), "--output-dir", outd, "--backend", "cpu-ref",
        "--width", "16", "--height", "12", ok=0)
    br = json.load(open(os.path.join(outd, "batch_report.json")))
    check([f["frame_id"] for f in br["frames"]] == ["frame_9", "frame_10", "frame_2"], "batch order")
    check(br["counts"]["completed"] == 3 and br["counts"]["dropped"] == 0, "batch counts")
    check(br["reprepare_count"] == 2, "batch reprepare on shape change")
    mapping = [json.loads(l) for l in open(os.path.join(outd, "outputs.jsonl"))]
    check([m["output_path"] for m in mapping] == ["frame_9.png", "frame_10.png", "frame_2.png"], "mapping manifest")
    run("batch", "--manifest", os.path.join(bd, "m.jsonl"), "--output-dir", outd, "--backend", "cpu-ref", ok=3)
    with open(os.path.join(bd, "dup.jsonl"), "w") as f:
        f.write('{"frame_id":"a","path":"frame_9.png"}\n{"frame_id":"a","path":"frame_2.png"}\n')
    p = run("batch", "--manifest", os.path.join(bd, "dup.jsonl"), "--output-dir", os.path.join(T, "b2"), ok=2)
    check("duplicate" in p.stderr, "duplicate id rejected")
    open(os.path.join(bd, "broken.png"), "wb").write(b"junk")
    with open(os.path.join(bd, "err.jsonl"), "w") as f:
        f.write('{"frame_id":"a","path":"frame_9.png"}\n{"frame_id":"b","path":"broken.png"}\n'
                '{"frame_id":"c","path":"frame_2.png"}\n')
    run("batch", "--manifest", os.path.join(bd, "err.jsonl"), "--output-dir", os.path.join(T, "b3"),
        "--backend", "cpu-ref", "--width", "8", "--height", "8", ok=5)
    b3 = json.load(open(os.path.join(T, "b3", "batch_report.json")))
    check(b3["counts"] == {"submitted": 2, "completed": 1, "failed": 1, "not_processed": 1, "dropped": 0},
          f"fail-fast counts {b3['counts']}")
    run("batch", "--manifest", os.path.join(bd, "err.jsonl"), "--output-dir", os.path.join(T, "b4"),
        "--backend", "cpu-ref", "--width", "8", "--height", "8", "--continue-on-error", ok=5)
    b4 = json.load(open(os.path.join(T, "b4", "batch_report.json")))
    check(b4["counts"]["completed"] == 2 and b4["counts"]["failed"] == 1, "continue-on-error counts")
    check(b4["throughput_completed_fps"] is None or b4["throughput_completed_fps"] > 0, "throughput from completed")
    # Explicit serial path vs parallel path: same outputs, policy recorded.
    outs = os.path.join(T, "b5s")
    run("batch", "--manifest", os.path.join(bd, "m.jsonl"), "--output-dir", outs, "--backend", "cpu-ref",
        "--width", "16", "--height", "12", "--decode-threads", "1", "--sync-write", ok=0)
    b5s = json.load(open(os.path.join(outs, "batch_report.json")))
    check(b5s["policy"]["decode_threads"] == 1 and b5s["policy"]["async_write"] is False, f"policy {b5s['policy']}")
    check(br["policy"]["async_write"] is True and br["policy"]["decode_threads"] >= 1, f"defaults {br['policy']}")
    check(br["policy"]["queue_capacity"] >= br["policy"]["decode_threads"], "default window covers decoders")
    outp = os.path.join(T, "b5")
    run("batch", "--manifest", os.path.join(bd, "m.jsonl"), "--output-dir", outp, "--backend", "cpu-ref",
        "--width", "16", "--height", "12", "--decode-threads", "3", ok=0)
    b5 = json.load(open(os.path.join(outp, "batch_report.json")))
    check(b5["policy"]["decode_threads"] == 3 and b5["policy"]["queue_capacity"] == 3, f"policy {b5['policy']}")
    check([f["frame_id"] for f in b5["frames"]] == ["frame_9", "frame_10", "frame_2"], "parallel batch order")
    for name in ("frame_9.png", "frame_10.png", "frame_2.png"):
        same = open(os.path.join(outs, name), "rb").read() == open(os.path.join(outp, name), "rb").read()
        check(same, f"parallel batch output {name} matches serial")
    for bad in ("0", "17", "x"):
        run("batch", "--manifest", os.path.join(bd, "m.jsonl"), "--output-dir", os.path.join(T, "b6"),
            "--decode-threads", bad, ok=2)
finally:
    shutil.rmtree(T, ignore_errors=True)

if failures:
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    print(f"{len(failures)} CLI check(s) failed", file=sys.stderr)
    sys.exit(1)
print("CLI integration checks passed", file=sys.stderr)
