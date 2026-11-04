#include "app/batch.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

#include "app/bounded_queue.hpp"
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

    BoundedQueue<Decoded> queue(opt.queue_capacity);
    std::atomic<bool> cancel{false};
    const auto t0 = clk::now();
    const int max_dim = opt.config.resources.max_dimension;

    std::thread producer([&] {
        for (std::size_t i = 0; i < manifest.entries.size() && !cancel.load(); ++i) {
            Decoded d;
            d.sequence = static_cast<std::int64_t>(i);
            d.accepted_ms = ms(t0);
            try {
                d.image = io::read_image(manifest.entries[i].resolved_path, max_dim);
                if (opt.on_decoded) opt.on_decoded(manifest.entries[i].frame_id);
            } catch (const Error& e) {
                d.error = e.what();
                d.category = e.category();
            } catch (const std::exception& e) {
                d.error = e.what();
            }
            if (!queue.push(std::move(d))) break;  // blocks while full: backpressure
        }
        queue.close();
    });

    HostFloatImage out(opt.config.resize.width, opt.config.resize.height);
    std::optional<InputSpec> prepared;
    std::int64_t expected = 0;
    try {
        while (auto item = queue.pop()) {
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
                const std::string rel = rec.frame_id + ".png";
                io::write_png_gray((fs::path(opt.output_dir) / rel).string(), io::quantize_image(out.cview()));
                rec.output_path = rel;
                rec.status = "completed";
                rec.completed_ms = ms(t0);
                ++res.completed;
            } catch (const Error& e) {
                rec.status = "failed";
                rec.error = e.what();
                rec.completed_ms = ms(t0);
                ++res.failed;
                if (!opt.continue_on_error) {
                    res.fail_fast_triggered = true;
                    cancel.store(true);
                    queue.clear();
                    queue.close();
                    break;
                }
            }
        }
    } catch (...) {
        cancel.store(true);
        queue.clear();
        queue.close();
        producer.join();
        throw;
    }
    producer.join();
    res.batch_wall_ms = ms(t0);
    res.queue_high_water = queue.high_water_mark();
    for (const auto& f : res.frames) {
        if (f.status == "not_processed") ++res.not_processed;
    }
    return res;
}

}  // namespace framepipe::app
