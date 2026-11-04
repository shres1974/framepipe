#pragma once
// OpenMP row partitioning for the cpu-omp backend. Only compiled when OpenMP
// is available; the serial reference never depends on it.

#include <functional>

namespace framepipe::cpu {

// Runs fn(y_begin, y_end) over disjoint contiguous row chunks covering
// [0, rows) using exactly `threads` OpenMP threads. Each output row is written
// by one thread, so results are identical to the serial reference.
void parallel_rows(int threads, int rows, const std::function<void(int, int)>& fn);

int openmp_max_threads() noexcept;
// "dynamic4x" (default) or "static" (FRAMEPIPE_OMP_STATIC_SCHEDULE=ON baseline).
const char* schedule_name() noexcept;
const char* openmp_version_string() noexcept;

}  // namespace framepipe::cpu
