#pragma once
// Offline ordered sequence runner (spec Section 15, milestone M6 core).
//
// Policy: blocking backpressure (no frames are dropped), manifest order is
// authoritative, fail-fast by default. With continue_on_error every failed
// frame is recorded and excluded from completed throughput. A shape change
// drains outstanding work and re-prepares (prepare() is synchronous).
//
// Threads: `decode_threads` decoder threads read/decode frames (possibly out
// of order) into a bounded reorder window of capacity `queue_capacity`; the
// calling thread takes them in sequence order, processes, quantizes, and
// writes them. Frames held by the window (being decoded or decoded and
// waiting) never exceed queue_capacity, so retained decoded frames are
// bounded by queue_capacity + 1 regardless of the decoder count. With
// async_write, a writer thread PNG-encodes and writes frames from a second
// bounded FIFO (also queue_capacity); a frame's completed_ms is when it
// reached its final status.

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "framepipe/config.hpp"
#include "io/manifest.hpp"

namespace framepipe::app {

struct BatchOptions {
    PipelineConfig config;
    std::string output_dir;
    bool overwrite = false;
    bool continue_on_error = false;
    std::size_t queue_capacity = 2;
    // Number of decoder threads (>= 1). Decoding is the largest per-frame
    // cost for large PNG inputs, so more than one decoder raises throughput.
    int decode_threads = 1;
    // Encode and write PNGs on a dedicated writer thread (FIFO, so outputs
    // still complete in manifest order) instead of on the processing thread.
    bool async_write = false;
    // Test hook: called on the decoder thread after each successful decode.
    std::function<void(const std::string& frame_id)> on_decoded;
};

struct FrameRecord {
    std::int64_t sequence = 0;
    std::string frame_id;
    std::string input_path;
    std::string output_path;  // relative to output_dir; empty if failed
    std::optional<std::int64_t> source_timestamp_ns;
    std::string status;       // "completed", "failed", "not_processed"
    std::string error;
    int input_width = 0, input_height = 0;
    double accepted_ms = 0, completed_ms = 0;  // monotonic, relative to batch start
};

struct BatchResult {
    std::vector<FrameRecord> frames;  // in manifest order
    std::size_t submitted = 0, completed = 0, failed = 0, not_processed = 0;
    std::size_t reprepare_count = 0;
    std::size_t queue_high_water = 0;
    double batch_wall_ms = 0;
    std::string selected_backend, implementation, selection_reason;
    bool fail_fast_triggered = false;
};

BatchResult run_batch(const io::Manifest& manifest, const BatchOptions& opt);

}  // namespace framepipe::app
