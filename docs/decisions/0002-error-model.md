# 0002 Error model: typed exceptions

Status: accepted, 2026-11-03

The public C++17 API reports failures with one mechanism:
`framepipe::Error` (derived from `std::runtime_error`) carrying an
`ErrorCategory` and an `ErrorContext {operation, backend, shape, detail}`.
`std::expected` is C++23 and is not used.

Categories: invalid_argument, invalid_configuration, invalid_image,
unsupported_format, backend_unavailable, allocation_failure, device_failure,
io_failure, correctness_failure, internal_error.

Rules:
- Destructors never throw; resource owners are move-only.
- `prepare` builds into locals and commits on success, so a failure halfway
  through allocation leaves no partial state (tested with an allocation
  fault-injection hook, `framepipe::testing::fail_host_allocation_after`).
- CUDA out-of-memory maps to `allocation_failure`, other CUDA errors to
  `device_failure`, always tagged with the stage name. Output is never
  returned after a failed completion.
- User-facing messages never include raw pointer values.

## Exit codes (spec Section 14 suggestion, adopted unchanged)

| Code | Meaning | Categories |
|---|---|---|
| 0 | success | — |
| 2 | invalid arguments or configuration | invalid_argument, invalid_configuration |
| 3 | I/O failure | io_failure, unsupported_format |
| 4 | backend unavailable | backend_unavailable |
| 5 | processing or resource failure | invalid_image, allocation_failure, device_failure, internal_error |
| 6 | correctness failure | correctness_failure (and `validate` mismatches) |

Unsupported formats map to 3 because they are discovered while reading the
input file. `doctor` returns 0 on a CPU-only machine.
