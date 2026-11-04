# Results

Benchmark output is **not committed** by default (`results/local/` is
git-ignored). Each `framepipe benchmark` run writes a new directory with
`run.json`, `environment.json`, `resolved_config.json`, `inputs.jsonl`,
`validation.json`, `samples.jsonl`, `summary.json`, `summary.csv`,
`analysis.md`. See [docs/benchmark_method.md](../docs/benchmark_method.md).

To publish a run as evidence, copy the whole directory (not only the
summary) to `results/evidence/<date>_<host>_<scope>/`, run
`python3 scripts/validate_results.py` on it, and reference it from
[docs/benchmark_results.md](../docs/benchmark_results.md).

Never hand-edit result files. Never create measured charts from a `not_run`
report (`scripts/analyze_results.py` refuses).
