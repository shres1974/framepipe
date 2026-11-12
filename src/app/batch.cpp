#include "app/batch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

#include "app/bounded_queue.hpp"
#include "app/ordered_window.hpp"
#include "framepipe/errors.hpp"
#include "framepipe/pipeline.hpp"
#include "io/image_io.hpp"

namespace framepipe::app {

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

namespace {

struct Decoded {
    std::int64_t sequence = 0;
    std::optional<io::DecodedImage> image;  // empty => decode failed
    std::string error;
    ErrorCategory category = ErrorCategory::IoFailure;
    double accepted_ms = 0;
};

// A processed, quantized frame waiting to be PNG-encoded and written.
struct Encoded {
    std::int64_t sequence;
    HostU8Image image;
};

double ms(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

}  // namespace

BatchResult run_batch(const io::Manifest& manifest, const BatchOptions& opt) {
    BatchResult res;
    Pipeline pipeline(opt.config);
    res.selected_backend = backend_name(pipeline.selected_backend());
    res.implementation = pipeline.configuration().implementation;
    res.selection_reason = pipeline.configuration().selection_reason;

    std::error_code ec;
    fs::create_directories(opt.output_dir, ec);
    if (ec) throw Error(ErrorCategory::IoFailure, "cannot create output directory: " + opt.output_dir, {"", "", "", ec.message()});

    // Pre-compute output paths; frame IDs are validated safe by the manifest
    // loader so they cannot escape output_dir.
    res.frames.resize(manifest.entries.size());
    for (std::size_t i = 0; i < manifest.entries.size(); ++i) {
        const auto& e = manifest.entries[i];
        auto& r = res.frames[i];
        r.sequence = static_cast<std::int64_t>(i);
        r.frame_id = e.frame_id;
        r.input_path = e.path;
        r.source_timestamp_ns = e.source_timestamp_ns;
        r.status = "not_processed";
        const fs::path outp = fs::path(opt.output_dir) / (e.frame_id + ".png");
        if (!opt.overwrite && fs::exists(outp, ec)) {
            throw Error(ErrorCategory::IoFailure, "refusing to overwrite existing output (use --overwrite): " + outp.string());
        }
        if (fs::exists(e.resolved_path, ec) && fs::exists(outp, ec) && fs::equivalent(e.resolved_path, outp, ec)) {
            throw Error(ErrorCategory::InvalidArgument, "output path resolves to the input file: " + outp.string());
        }
    }

    const auto total = static_cast<std::int64_t>(manifest.entries.size());
    OrderedWindow<Decoded> window(opt.queue_capacity, total);
    std::atomic<std::int64_t> next_claim{0};
    const auto t0 = clk::now();
    const int max_dim = opt.config.resources.max_dimension;

    // Decoder workers claim sequence numbers in increasing order and may
    // finish out of order; the window hands frames to this thread in order.
    auto decode_worker = [&] {
        for (std::int64_t i = next_claim++; i < total; i = next_claim++) {
            if (!window.acquire(i)) return;  // blocks while the window is full: backpressure
            const auto& entry = manifest.entries[static_cast<std::size_t>(i)];
            Decoded d;
            d.sequence = i;
            d.accepted_ms = ms(t0);
            try {
                d.image = io::read_image(entry.resolved_path, max_dim);
                if (opt.on_decoded) opt.on_decoded(entry.frame_id);
            } catch (const Error& e) {
                d.error = e.what();
                d.category = e.category();
            } catch (const std::exception& e) {
                d.error = e.what();
            }
            window.put(i, std::move(d));
        }
    };
    const int n_decoders = std::max(1, std::min(opt.decode_threads, static_cast<int>(std::max<std::int64_t>(total, 1))));
    std::vector<std::thread> decoders;
    decoders.reserve(static_cast<std::size_t>(n_decoders));
    for (int t = 0; t < n_decoders; ++t) decoders.emplace_back(decode_worker);
    auto stop_decoders = [&] {
        window.close();
        for (auto& th : decoders) th.join();
    };

    // Final per-frame status is set either here (processing failures, or
    // synchronous writes) or on the writer thread, so it is guarded.
    std::mutex rec_m;
    auto finish = [&](std::int64_t seq, const std::string& rel, const std::string* err) {
        std::lock_guard<std::mutex> lk(rec_m);
        auto& rec = res.frames[static_cast<std::size_t>(seq)];
        rec.completed_ms = ms(t0);
        if (err) {
            rec.status = "failed";
            rec.error = *err;
            ++res.failed;
            if (!opt.continue_on_error) res.fail_fast_triggered = true;
        } else {
            rec.output_path = rel;
            rec.status = "completed";
            ++res.completed;
        }
    };
    std::atomic<bool> write_failed{false};
    auto write_frame = [&](Encoded& e) {
        const std::string rel = res.frames[static_cast<std::size_t>(e.sequence)].frame_id + ".png";
        std::string err;
        try {
            io::write_png_gray((fs::path(opt.output_dir) / rel).string(), e.image);
            finish(e.sequence, rel, nullptr);
            return;
        } catch (const Error& x) {
            err = x.what();
        } catch (const std::exception& x) {
            err = x.what();
        }
        finish(e.sequence, "", &err);
        if (!opt.continue_on_error) write_failed = true;
    };
    const bool fail_fast = !opt.continue_on_error;

    // Writer thread: PNG encode + file write overlap with processing the next
    // frame. It consumes in FIFO (= manifest) order, so completion order is
    // preserved. After a fail-fast write failure it drains without writing.
    BoundedQueue<Encoded> to_write(opt.queue_capacity);
    std::thread writer;
    if (opt.async_write) {
        writer = std::thread([&] {
            while (auto e = to_write.pop()) {
                if (fail_fast && write_failed) continue;
                write_frame(*e);
            }
        });
    }
    auto stop_writer = [&] {
        to_write.close();
        if (writer.joinable()) writer.join();
    };

    HostFloatImage out(opt.config.resize.width, opt.config.resize.height);
    std::optional<InputSpec> prepared;
    std::int64_t expected = 0;
    try {
        while (auto item = window.take()) {
            if (fail_fast && write_failed) break;
            if (item->sequence != expected) {
                throw Error(ErrorCategory::Internal, "sequence order violated in batch runner");
            }
            ++expected;
            ++res.submitted;
            auto& rec = res.frames[static_cast<std::size_t>(item->sequence)];
            rec.accepted_ms = item->accepted_ms;
            try {
                if (!item->image) throw Error(item->category, item->error);
                const auto view = item->image->image.view();
                rec.input_width = view.width;
                rec.input_height = view.height;
                const InputSpec spec = spec_of(view);
                if (!prepared || *prepared != spec) {
                    if (prepared) ++res.reprepare_count;
                    pipeline.prepare(spec);  // synchronous: owned work already drained
                    prepared = spec;
                }
                pipeline.process_into(view, out.view());
                Encoded e{item->sequence, io::quantize_image(out.cview())};
                if (opt.async_write) {
                    if (!to_write.push(std::move(e))) break;  // blocks while the writer is behind
                } else {
                    write_frame(e);
                }
            } catch (const Error& e) {
                const std::string msg = e.what();
                finish(item->sequence, "", &msg);
                if (fail_fast) break;
            }
            if (fail_fast && write_failed) break;
        }
    } catch (...) {
        to_write.clear();
        stop_writer();
        stop_decoders();
        throw;
    }
    stop_writer();
    stop_decoders();
    res.batch_wall_ms = ms(t0);
    res.queue_high_water = window.high_water_mark();
    for (const auto& f : res.frames) {
        if (f.status == "not_processed") ++res.not_processed;
    }
    return res;
}

}  // namespace framepipe::app
