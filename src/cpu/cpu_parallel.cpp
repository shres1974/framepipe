#include "cpu/cpu_parallel.hpp"

#include <omp.h>

#include <exception>
#include <mutex>

namespace framepipe::cpu {

void parallel_rows(int threads, int rows, const std::function<void(int, int)>& fn) {
    if (threads <= 1 || rows <= 1) {
        fn(0, rows);
        return;
    }
    std::exception_ptr first_error;
    std::mutex m;
    auto run_chunk = [&](int c, int chunks) {
        const int b = static_cast<int>(static_cast<long long>(rows) * c / chunks);
        const int e = static_cast<int>(static_cast<long long>(rows) * (c + 1) / chunks);
        try {
            fn(b, e);
        } catch (...) {
            std::lock_guard<std::mutex> lk(m);
            if (!first_error) first_error = std::current_exception();
        }
    };
#if FRAMEPIPE_OMP_STATIC_SCHEDULE
    // Baseline variant (kept for comparison): one contiguous chunk per thread.
    const int chunks = threads < rows ? threads : rows;
#pragma omp parallel for num_threads(chunks) schedule(static, 1)
    for (int c = 0; c < chunks; ++c) run_chunk(c, chunks);
#else
    // Default: several chunks per thread with dynamic scheduling so a slow or
    // descheduled core (hybrid P/E cores, VMs) does not stall the stage
    // barrier. Each row is still written by exactly one thread.
    const int target = threads * 4;
    const int chunks = target < rows ? target : rows;
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int c = 0; c < chunks; ++c) run_chunk(c, chunks);
#endif
    if (first_error) std::rethrow_exception(first_error);
}

const char* schedule_name() noexcept { return FRAMEPIPE_OMP_STATIC_SCHEDULE ? "static" : "dynamic4x"; }

int openmp_max_threads() noexcept { return omp_get_max_threads(); }

const char* openmp_version_string() noexcept {
    // _OPENMP is yyyymm of the supported specification.
#define FP_STR2(x) #x
#define FP_STR(x) FP_STR2(x)
    return FP_STR(_OPENMP);
#undef FP_STR
#undef FP_STR2
}

}  // namespace framepipe::cpu
