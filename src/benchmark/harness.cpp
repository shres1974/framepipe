#include "benchmark/harness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>

#include <opencv2/core.hpp>

#include "benchmark/environment.hpp"
#include "benchmark/opencv_baseline.hpp"
#include "benchmark/stats.hpp"
#include "core/sha256.hpp"
#include "framepipe/errors.hpp"
#include "framepipe/pipeline.hpp"
#include "io/image_io.hpp"
#include "io/json_util.hpp"
#include "io/manifest.hpp"
#include "validation/compare.hpp"

namespace framepipe::bench {

namespace fs = std::filesystem;
using ojson = nlohmann::ordered_json;
using clk = std::chrono::steady_clock;

namespace {

double ms_since(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Consumes output outside timing so measured work cannot be discarded.
volatile double g_sink = 0.0;
double consume(const HostFloatImage& im) {
    double s = 0;
    for (int y = 0; y < im.height(); y += std::max(1, im.height() / 16)) {
        for (int x = 0; x < im.width(); ++x) s += im.at(x, y);
    }
    g_sink = g_sink + s;
    return s;
}

struct Frame {
    io::ManifestEntry entry;
    HostU8Image image;
    std::string file_sha256;
    std::string source_format;
};

struct Sample {
    std::string backend, implementation, metric, scope, frame_id, workload;
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    int repetition = 0, iteration = 0;
    double duration_ms = 0;
    bool success = true;
    ojson extra = ojson::object();
};

ojson sample_json(const Sample& s, const std::string& run_id, const std::string& hash) {
    ojson j;
    j["run_id"] = run_id;
    j["backend"] = s.backend;
    j["implementation"] = s.implementation;
    j["metric"] = s.metric;
    j["scope"] = s.scope;
    j["workload"] = s.workload;
    j["frame_id"] = s.frame_id;
    j["input_width"] = s.in_w;
    j["input_height"] = s.in_h;
    j["output_width"] = s.out_w;
    j["output_height"] = s.out_h;
    j["repetition"] = s.repetition;
    j["iteration"] = s.iteration;
    j["duration_ms"] = s.duration_ms;
    j["success"] = s.success;
    j["math_config_hash"] = hash;
    for (auto it = s.extra.begin(); it != s.extra.end(); ++it) j[it.key()] = it.value();
    return j;
}

// A measurable thing: a FramePipe pipeline or the OpenCV baseline.
struct Runner {
    std::string name;  // cpu-ref, cpu-omp, cuda, opencv-cpu
    std::string implementation;
    std::unique_ptr<Pipeline> pipeline;
    std::unique_ptr<OpenCvBaseline> opencv;
    std::optional<InputSpec> prepared;
    std::string status = "ok";  // ok, unavailable, failed_correctness, failed_runtime
    std::string reason;
    bool is_gpu() const { return name == "cuda"; }
    void prepare(const InputSpec& s) {
        if (prepared && *prepared == s) return;
        if (pipeline) pipeline->prepare(s);
        if (opencv) opencv->prepare(s);
        prepared = s;
    }
    void process(const InputImageView& in, const FloatImageView& out) {
        if (pipeline) pipeline->process_into(in, out);
        else opencv->process_into(in, out);
    }
};

PipelineConfig config_for(const PipelineConfig& base, const std::string& backend, int omp_threads) {
    PipelineConfig c = base;
    if (backend == "cpu-ref") c.backend = Backend::CpuReference;
    else if (backend == "cpu-omp") {
        c.backend = Backend::CpuParallel;
        if (omp_threads > 0) c.cpu.threads = omp_threads;
    } else if (backend == "cuda") c.backend = Backend::Cuda;
    else c.backend = Backend::CpuReference;
    return c;
}

std::string metric_for(const std::string& scope, const Runner& r) {
    if (scope == "host_processing") return r.is_gpu() ? "gpu_transfer_inclusive_wall_ms" : "cpu_core_wall_ms";
    if (scope == "device_pipeline") return "gpu_device_pipeline_ms";
    if (scope == "application") return "application_wall_ms";
    if (scope == "startup") return "startup_wall_ms";
    if (scope == "batch") return "batch_wall_ms";
    return "unknown";
}

std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string r = "\"";
    for (char c : s) {
        if (c == '"') r += "\"\"";
        else r += c;
    }
    return r + "\"";
}

std::string fmt(double v, int prec = 4) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(prec);
    os << v;
    return os.str();
}

}  // namespace

int run_benchmark(const BenchmarkOptions& opt) {
    // ------------------------------------------------------------ validate options
    const std::vector<std::string> known_backends{"cpu-ref", "cpu-omp", "cuda", "opencv-cpu"};
    const std::vector<std::string> known_scopes{"host_processing", "device_pipeline", "application", "startup", "batch"};
    for (const auto& b : opt.backends) {
        if (std::find(known_backends.begin(), known_backends.end(), b) == known_backends.end()) {
            throw Error(ErrorCategory::InvalidArgument,
                        "unknown benchmark backend '" + b + "' (accepted: cpu-ref, cpu-omp, cuda, opencv-cpu)");
        }
    }
    for (const auto& s : opt.scopes) {
        if (std::find(known_scopes.begin(), known_scopes.end(), s) == known_scopes.end()) {
            throw Error(ErrorCategory::InvalidArgument,
                        "unknown scope '" + s +
                            "' (accepted: host_processing, device_pipeline, application, startup, batch)");
        }
    }
    if (opt.backends.empty()) throw Error(ErrorCategory::InvalidArgument, "--backends must not be empty");
    if (opt.scopes.empty()) throw Error(ErrorCategory::InvalidArgument, "--scopes must not be empty");
    if (opt.warmup < 0 || opt.iterations < 1 || opt.repetitions < 1) {
        throw Error(ErrorCategory::InvalidArgument, "--warmup must be >= 0, --iterations and --repetitions >= 1");
    }
    if (opt.workload != "repeat" && opt.workload != "ring") {
        throw Error(ErrorCategory::InvalidArgument, "--workload must be 'repeat' or 'ring'");
    }
    if (opt.output_dir.empty()) throw Error(ErrorCategory::InvalidArgument, "--output-dir is required");

    PipelineConfig base;
    if (!opt.config_path.empty()) base = apply_config_json(io::read_text_file(opt.config_path), base);
    if (!opt.config_overrides_json.empty()) base = apply_config_json(opt.config_overrides_json, base);

    // ------------------------------------------------------------ result directory
    std::error_code ec;
    if (fs::exists(opt.output_dir, ec)) {
        if (!opt.overwrite) {
            throw Error(ErrorCategory::IoFailure,
                        "result directory already exists (use --overwrite or choose a new --output-dir): " +
                            opt.output_dir);
        }
    }
    fs::create_directories(fs::path(opt.output_dir) / "scratch", ec);
    if (ec) throw Error(ErrorCategory::IoFailure, "cannot create result directory: " + opt.output_dir, {"", "", "", ec.message()});
    const auto out = [&](const std::string& f) { return (fs::path(opt.output_dir) / f).string(); };

    const std::string run_id = io::make_run_id();
    const std::string started = io::utc_timestamp_iso8601();
    const GitInfo git = git_info();
    const std::string hash = math_config_hash(base);

    // ------------------------------------------------------------ inputs
    const io::Manifest manifest = io::load_manifest(opt.manifest_path, true);
    std::vector<Frame> frames;
    {
        std::ofstream inputs(out("inputs.jsonl"));
        for (const auto& e : manifest.entries) {
            Frame f;
            f.entry = e;
            f.file_sha256 = sha256_file_hex(e.resolved_path);
            auto d = io::read_image(e.resolved_path, base.resources.max_dimension);
            f.source_format = d.source_format;
            f.image = std::move(d.image);
            ojson j;
            j["frame_id"] = e.frame_id;
            j["path"] = e.path;
            j["width"] = f.image.width();
            j["height"] = f.image.height();
            j["format"] = to_string(f.image.format());
            j["decoded_from"] = f.source_format;
            j["sha256"] = f.file_sha256;
            j["source_timestamp_ns"] = e.source_timestamp_ns ? ojson(*e.source_timestamp_ns) : ojson(nullptr);
            j["source_attribution"] = "see manifest directory README / fixture generator metadata";
            inputs << j.dump() << "\n";
            frames.push_back(std::move(f));
        }
    }
    if (opt.workload == "ring") {
        for (const auto& f : frames) {
            if (f.image.width() != frames[0].image.width() || f.image.height() != frames[0].image.height() ||
                f.image.format() != frames[0].image.format()) {
                throw Error(ErrorCategory::InvalidArgument,
                            "ring workload requires all manifest frames to share one shape/format");
            }
        }
    }

    io::write_json_file(out("resolved_config.json"), ojson::parse(config_to_json(base)), true);
    ojson env = collect_environment(base.device_index);
    io::write_json_file(out("environment.json"), env, true);

    // ------------------------------------------------------------ backends
    std::vector<Runner> runners;
    for (const auto& b : opt.backends) {
        Runner r;
        r.name = b;
        try {
            if (b == "opencv-cpu") {
                r.opencv = std::make_unique<OpenCvBaseline>(base, opt.opencv_threads);
                r.implementation = r.opencv->implementation();
            } else {
                r.pipeline = std::make_unique<Pipeline>(config_for(base, b, opt.omp_threads));
                r.implementation = r.pipeline->configuration().implementation;
            }
        } catch (const Error& e) {
            r.status = "unavailable";
            r.reason = e.what();
            std::cerr << "benchmark: backend " << b << " unavailable: " << e.what() << "\n";
        }
        runners.push_back(std::move(r));
    }

    // ------------------------------------------------------------ correctness gate
    // Every candidate is compared against cpu-ref per frame before measuring.
    // The cpu-ref comparator is itself validated by the independent oracle suite.
    ojson validation;
    validation["schema_version"] = 1;
    validation["reference"] = "cpu-ref";
    validation["note"] =
        "cpu-ref correctness is established by the independent NumPy oracle and hand fixtures (ctest); "
        "this file records candidate-vs-cpu-ref parity for the benchmarked inputs.";
    validation["output_stage"] = to_string(base.output_stage);
    ojson vb = ojson::array();
    {
        PipelineConfig rc = config_for(base, "cpu-ref", 0);
        Pipeline ref(rc);
        for (auto& r : runners) {
            ojson entry;
            entry["backend"] = r.name;
            entry["implementation"] = r.implementation;
            if (r.status != "ok") {
                entry["status"] = "not_executed";
                entry["reason"] = r.reason;
                vb.push_back(entry);
                continue;
            }
            if (r.name == "cpu-ref") {
                entry["status"] = "reference";
                vb.push_back(entry);
                continue;
            }
            const Stage final_stage = base.output_stage == OutputStage::Edges  ? Stage::Edges
                                      : base.output_stage == OutputStage::Blur ? Stage::BlurVertical
                                                                               : Stage::Resized;
            const Tolerance tol = tolerance_for(final_stage);
            bool all_pass = true;
            ojson per = ojson::array();
            try {
                for (auto& f : frames) {
                    const auto spec = spec_of(f.image.view());
                    ref.prepare(spec);
                    r.prepare(spec);
                    HostFloatImage a(base.resize.width, base.resize.height), b(base.resize.width, base.resize.height);
                    ref.process_into(f.image.view(), a.view());
                    r.process(f.image.view(), b.view());
                    const auto cr = compare_float_images(a.cview(), b.cview(), tol, base.output_stage == OutputStage::Edges);
                    ojson fj = ojson::parse(compare_result_json(cr));
                    fj["frame_id"] = f.entry.frame_id;
                    per.push_back(fj);
                    all_pass = all_pass && cr.passed;
                }
            } catch (const Error& e) {
                r.status = "failed_runtime";
                r.reason = e.what();
                entry["status"] = "failed";
                entry["reason"] = e.what();
                entry["frames"] = per;
                vb.push_back(entry);
                continue;
            }
            entry["status"] = all_pass ? "passed" : "failed";
            entry["frames"] = per;
            if (!all_pass) {
                if (r.name == "opencv-cpu") {
                    // Not a FramePipe backend: still measured, but labeled as a
                    // separate workload with different semantics.
                    entry["matched_semantics"] = false;
                    entry["note"] = "OpenCV output differs from the FramePipe contract beyond tolerance; its timings "
                                    "are a separately labeled practical-library workload, not a matched comparison.";
                } else {
                    r.status = "failed_correctness";
                    r.reason = "output differs from cpu-ref beyond tolerance";
                }
            } else if (r.name == "opencv-cpu") {
                entry["matched_semantics"] = true;
            }
            vb.push_back(entry);
        }
    }
    validation["backends"] = vb;
    io::write_json_file(out("validation.json"), validation, true);

    // ------------------------------------------------------------ measurement
    std::vector<Sample> samples;
    ojson run_order = ojson::array();
    ojson scope_status = ojson::array();
    std::map<std::string, bool> cold_seen;
    long long zero_samples = 0;
    const int wo = base.resize.width, ho = base.resize.height;
    HostFloatImage out_img(wo, ho);

    auto add = [&](Sample s) {
        if (s.duration_ms <= 0.0) ++zero_samples;
        samples.push_back(std::move(s));
    };

    for (int rep = 0; rep < opt.repetitions; ++rep) {
        // Deterministic rotation of backend order across repetitions.
        std::vector<std::size_t> order(runners.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = (i + static_cast<std::size_t>(rep)) % order.size();
        ojson ro = ojson::array();
        for (auto i : order) ro.push_back(runners[i].name);
        run_order.push_back({{"repetition", rep}, {"order", ro}});

        for (auto idx : order) {
            Runner& r = runners[idx];
            if (r.status != "ok") continue;
            try {
                for (const auto& scope : opt.scopes) {
                    const std::string metric = metric_for(scope, r);
                    if (scope == "device_pipeline" && !(r.pipeline && r.pipeline->supports_device_pipeline_timing())) {
                        continue;  // not synthesized for CPU backends
                    }
                    if ((scope == "startup") && !r.pipeline) continue;

                    if (scope == "startup") {
                        const auto spec = spec_of(frames[0].image.view());
                        const bool cold = !cold_seen[r.name];
                        const auto t0 = clk::now();
                        Pipeline p(config_for(base, r.name, opt.omp_threads));
                        p.prepare(spec);
                        const auto t1 = clk::now();
                        cold_seen[r.name] = true;
                        Sample s{r.name, r.implementation, metric, scope, frames[0].entry.frame_id, opt.workload,
                                 spec.width, spec.height, wo, ho, rep, 0, ms_since(t0, t1)};
                        s.extra["process_cold"] = cold;
                        add(std::move(s));
                        continue;
                    }

                    if (scope == "batch") {
                        // Whole manifest in order: read, decode, prepare-if-needed,
                        // process, quantize, encode. One sample per repetition.
                        const auto scratch = (fs::path(opt.output_dir) / "scratch" / (r.name + "_batch.png")).string();
                        const auto t0 = clk::now();
                        for (const auto& f : frames) {
                            auto d = io::read_image(f.entry.resolved_path, base.resources.max_dimension);
                            r.prepare(spec_of(d.image.view()));
                            r.process(d.image.view(), out_img.view());
                            io::write_png_gray(scratch, io::quantize_image(out_img.cview()));
                        }
                        const auto t1 = clk::now();
                        consume(out_img);
                        Sample s{r.name, r.implementation, metric, scope, "__batch__", "batch",
                                 0, 0, wo, ho, rep, 0, ms_since(t0, t1)};
                        s.extra["completed_frames"] = static_cast<long long>(frames.size());
                        s.extra["failed_frames"] = 0;
                        add(std::move(s));
                        continue;
                    }

                    // Per-frame scopes.
                    const std::size_t n_iter_frames = opt.workload == "ring" ? 1 : frames.size();
                    for (std::size_t fi = 0; fi < n_iter_frames; ++fi) {
                        auto frame_at = [&](int it) -> Frame& {
                            if (opt.workload == "ring") return frames[static_cast<std::size_t>(it) % frames.size()];
                            return frames[fi];
                        };
                        r.prepare(spec_of(frame_at(0).image.view()));
                        const auto scratch =
                            (fs::path(opt.output_dir) / "scratch" / (r.name + "_app.png")).string();
                        auto one = [&](int it, bool record) {
                            Frame& f = frame_at(it);
                            const auto iv = f.image.view();
                            double ms = 0;
                            Sample s{r.name, r.implementation, metric, scope, f.entry.frame_id, opt.workload,
                                     f.image.width(), f.image.height(), wo, ho, rep, it, 0};
                            if (scope == "host_processing") {
                                const auto t0 = clk::now();
                                r.process(iv, out_img.view());
                                const auto t1 = clk::now();
                                ms = ms_since(t0, t1);
                            } else if (scope == "device_pipeline") {
                                r.pipeline->device_resident_upload(iv);  // outside the interval
                                ms = r.pipeline->device_resident_run_timed_ms();
                                r.pipeline->device_resident_download(out_img.view());  // outside the interval
                                s.extra["label"] = "gpu-resident input; excludes H2D/D2H";
                            } else if (scope == "application") {
                                const auto t0 = clk::now();
                                const std::string bytes = io::read_file_bytes(f.entry.resolved_path);
                                auto d = io::decode_image(bytes, f.entry.resolved_path, base.resources.max_dimension);
                                const auto sp = spec_of(d.image.view());
                                const bool need_prep = !(r.prepared && *r.prepared == sp);
                                r.prepare(sp);
                                r.process(d.image.view(), out_img.view());
                                io::write_png_gray(scratch, io::quantize_image(out_img.cview()));
                                const auto t1 = clk::now();
                                ms = ms_since(t0, t1);
                                s.extra["process_state"] = "warm (backend initialized)";
                                s.extra["preparation_needed"] = need_prep;
                                s.extra["fsync"] = false;
                            }
                            consume(out_img);
                            if (record) {
                                s.duration_ms = ms;
                                add(std::move(s));
                            }
                        };
                        for (int w = 0; w < opt.warmup; ++w) one(w, false);
                        for (int it = 0; it < opt.iterations; ++it) one(it, true);
                    }
                }
            } catch (const Error& e) {
                r.status = "failed_runtime";
                r.reason = e.what();
                std::cerr << "benchmark: backend " << r.name << " failed: " << e.what() << "\n";
                scope_status.push_back({{"backend", r.name}, {"repetition", rep}, {"status", "failed"}, {"error", e.what()}});
                // Mark this backend's samples failed: excluded from success-only aggregates.
                for (auto& s : samples) {
                    if (s.backend == r.name) s.success = false;
                }
            }
        }
    }

    // ------------------------------------------------------------ samples.jsonl
    {
        std::ofstream sf(out("samples.jsonl"));
        for (const auto& s : samples) {
            auto j = sample_json(s, run_id, hash);
            io::require_finite_numbers(j, "samples");
            sf << j.dump() << "\n";
        }
    }

    // ------------------------------------------------------------ summary
    struct Key {
        std::string backend, implementation, metric, frame_id;
        bool operator<(const Key& o) const {
            return std::tie(backend, implementation, metric, frame_id) <
                   std::tie(o.backend, o.implementation, o.metric, o.frame_id);
        }
    };
    std::map<Key, std::map<int, std::vector<double>>> groups;
    std::map<Key, std::pair<int, int>> shapes;
    for (const auto& s : samples) {
        if (!s.success) continue;
        Key k{s.backend, s.implementation, s.metric, s.frame_id};
        groups[k][s.repetition].push_back(s.duration_ms);
        shapes[k] = {s.in_w, s.in_h};
    }
    ojson groups_j = ojson::array();
    std::map<Key, double> pooled_median;
    std::ofstream csv(out("summary.csv"));
    csv << "backend,implementation,metric,frame_id,input_width,input_height,output_width,output_height,count,min_ms,"
           "max_ms,mean_ms,stddev_ms,median_ms,p95_ms,median_of_run_medians_ms,repetitions\n";
    for (const auto& [k, by_rep] : groups) {
        std::vector<double> pooled;
        std::vector<double> run_medians;
        ojson per_run = ojson::array();
        for (const auto& [rep, v] : by_rep) {
            pooled.insert(pooled.end(), v.begin(), v.end());
            const Stats st = compute_stats(v);
            run_medians.push_back(st.median);
            ojson pr = stats_json(st);
            pr["repetition"] = rep;
            per_run.push_back(pr);
        }
        const Stats ps = compute_stats(pooled);
        pooled_median[k] = ps.median;
        ojson g;
        g["backend"] = k.backend;
        g["implementation"] = k.implementation;
        g["metric"] = k.metric;
        g["frame_id"] = k.frame_id;
        g["input_size"] = {shapes[k].first, shapes[k].second};
        g["output_size"] = {wo, ho};
        g["pooled_within_runs"] = stats_json(ps);
        g["median_of_run_medians_ms"] = median_of(run_medians);
        g["per_run"] = per_run;
        if (k.metric == "batch_wall_ms") {
            std::vector<double> fps;
            for (const auto& [rep, v] : by_rep) {
                for (double ms : v) fps.push_back(static_cast<double>(frames.size()) / (ms / 1000.0));
            }
            g["throughput_completed_fps_median"] = median_of(fps);
            g["throughput_definition"] = "completed frames / measured batch wall seconds, per repetition";
        }
        groups_j.push_back(g);
        const auto sd = ps.stddev ? fmt(*ps.stddev, 6) : std::string("");
        csv << csv_escape(k.backend) << "," << csv_escape(k.implementation) << "," << k.metric << ","
            << csv_escape(k.frame_id) << "," << shapes[k].first << "," << shapes[k].second << "," << wo << "," << ho
            << "," << ps.count << "," << fmt(ps.min, 6) << "," << fmt(ps.max, 6) << "," << fmt(ps.mean, 6) << "," << sd
            << "," << fmt(ps.median, 6) << "," << fmt(ps.p95, 6) << "," << fmt(median_of(run_medians), 6) << ","
            << by_rep.size() << "\n";
    }

    // Comparable ratios: like-for-like medians per frame.
    auto find_median = [&](const std::string& backend, const std::string& metric,
                           const std::string& frame) -> std::optional<double> {
        for (const auto& [k, v] : pooled_median) {
            if (k.backend == backend && k.metric == metric && k.frame_id == frame) return v;
        }
        return std::nullopt;
    };
    ojson ratios = ojson::array();
    for (const auto& f : frames) {
        const auto& id = f.entry.frame_id;
        const auto ref = find_median("cpu-ref", "cpu_core_wall_ms", id);
        const auto omp = find_median("cpu-omp", "cpu_core_wall_ms", id);
        const auto gpu_t = find_median("cuda", "gpu_transfer_inclusive_wall_ms", id);
        const auto gpu_d = find_median("cuda", "gpu_device_pipeline_ms", id);
        auto ratio = [](std::optional<double> a, std::optional<double> b) -> ojson {
            if (!a || !b || *b <= 0.0) return nullptr;
            return *a / *b;
        };
        ojson rj;
        rj["frame_id"] = id;
        rj["host_processing_speedup_vs_scalar_cpu_ref"] = ratio(ref, gpu_t);
        rj["host_processing_speedup_vs_cpu_omp"] = ratio(omp, gpu_t);
        rj["resident_processing_ratio_vs_scalar_cpu_ref"] = ratio(ref, gpu_d);
        rj["cpu_omp_speedup_vs_scalar_cpu_ref"] = ratio(ref, omp);
        rj["note"] =
            "Ratios of pooled medians. resident_processing_ratio describes GPU-resident processing and is NOT an "
            "application speedup. null means a required measurement was not taken.";
        ratios.push_back(rj);
    }

    ojson backends_j = ojson::array();
    bool any_measured = false, any_failed = false;
    for (const auto& r : runners) {
        ojson b;
        b["backend"] = r.name;
        b["implementation"] = r.implementation;
        b["status"] = r.status;
        b["reason"] = r.reason.empty() ? ojson(nullptr) : ojson(r.reason);
        if (r.name == "cpu-omp" && r.pipeline) b["threads"] = r.pipeline->configuration().config.cpu.threads;
        if (r.name == "opencv-cpu") b["threads"] = opt.opencv_threads;
        if (r.pipeline) {
            const auto w = r.pipeline->workspace();
            b["workspace"] = {{"host_logical_bytes", w.host_logical_bytes},
                              {"host_allocated_bytes", w.host_allocated_bytes},
                              {"device_logical_bytes", w.device_logical_bytes},
                              {"device_allocated_bytes", w.device_allocated_bytes},
                              {"pinned_host_bytes", w.pinned_host_bytes},
                              {"transfer_mode", w.transfer_mode}};
        }
        if (r.status == "unavailable" && r.name == "cuda") {
            b["reproduce"] =
                "cmake --preset cuda-release && cmake --build --preset cuda-release && build/cuda-release/bin/framepipe "
                "benchmark --manifest <manifest> --backends cpu-ref,cuda ... on a host with a usable NVIDIA GPU";
        }
        if (r.status == "ok") any_measured = true;
        if (r.status.rfind("failed", 0) == 0) any_failed = true;
        backends_j.push_back(b);
    }

    ojson summary;
    summary["schema_version"] = 1;
    summary["run_id"] = run_id;
    summary["status"] = any_measured ? (any_failed ? "partial" : "measured") : "not_run";
    summary["workload"] = opt.workload;
    summary["workload_note"] = opt.workload == "repeat"
                                   ? "Each frame processed repeatedly; repeated timings of one image are not "
                                     "independent observations of real-world image diversity."
                                   : "Frames cycled in manifest order (ring).";
    summary["counts"] = {{"warmup", opt.warmup}, {"iterations", opt.iterations}, {"repetitions", opt.repetitions}};
    summary["statistics_definitions"] = {
        {"median", "middle value, or mean of the two middle values"},
        {"p95", "nearest rank: sorted[ceil(0.95*N)-1]"},
        {"stddev", "sample standard deviation (N-1); null when N<2"},
        {"pooled_within_runs", "all samples of all repetitions pooled"},
        {"median_of_run_medians_ms", "median across repetitions of each repetition's median"}};
    summary["backends"] = backends_j;
    summary["groups"] = groups_j;
    summary["ratios"] = ratios;
    summary["zero_duration_samples"] = zero_samples;
    summary["math_config_hash"] = hash;
    io::write_json_file(out("summary.json"), summary, true);

    // ------------------------------------------------------------ run.json
    ojson run;
    run["schema_version"] = 1;
    run["run_id"] = run_id;
    run["started_utc"] = started;
    run["finished_utc"] = io::utc_timestamp_iso8601();
    run["command_line"] = opt.command_line;
    run["git_revision"] = git.known ? ojson(git.revision) : ojson(nullptr);
    run["git_dirty"] = git.known ? ojson(git.dirty) : ojson(nullptr);
    run["git_note"] = git.note.empty() ? ojson(nullptr) : ojson(git.note);
    run["experiment_status"] = summary["status"];
    run["manifest"] = {{"path", manifest.path}, {"sha256", manifest.sha256}, {"frames", frames.size()}};
    run["backend_run_order"] = run_order;
    run["failures"] = scope_status;
    run["scopes"] = opt.scopes;
    run["math_config_hash"] = hash;
    run["measurement_settings"] = {{"debug_sync", false},
                                   {"stage_downloads", false},
                                   {"sanitizers", env["build"]["sanitizers"]},
                                   {"per_frame_logging", false},
                                   {"clock_changes", "none performed"}};
    io::write_json_file(out("run.json"), run, true);

    // ------------------------------------------------------------ analysis.md
    {
        std::ostringstream md;
        md << "# Benchmark analysis: " << run_id << "\n\n";
        md << "Status: **" << summary["status"].get<std::string>() << "**. Generated by `framepipe benchmark`; "
           << "regenerate figures and re-derive statistics from `samples.jsonl` with "
           << "`python3 scripts/analyze_results.py " << opt.output_dir << "`.\n\n";
        md << "## Command\n\n```\n" << opt.command_line << "\n```\n\n";
        md << "## Hardware\n\n- CPU: " << (env["host"]["cpu_model"].is_null() ? "unknown" : env["host"]["cpu_model"].get<std::string>())
           << " (" << env["host"]["logical_cpus"].dump() << " logical CPUs)\n";
        md << "- GPU: ";
        if (env["cuda"]["devices"].empty()) md << "none measured (" << (env["cuda"]["error"].is_null() ? "n/a" : env["cuda"]["error"].get<std::string>()) << ")\n";
        else md << env["cuda"]["devices"][0]["name"].get<std::string>() << "\n";
        md << "- OS: " << (env["host"]["os"].is_null() ? "unknown" : env["host"]["os"].get<std::string>()) << ", "
           << env["host"]["kernel"].get<std::string>() << "\n";
        md << "- Build: " << env["build"]["build_type"].get<std::string>() << ", "
           << env["build"]["cxx_compiler"].get<std::string>() << ", flags `" << env["build"]["cxx_flags"].get<std::string>()
           << "`\n\n";
        md << "## Backends\n\n| backend | implementation | status | reason |\n|---|---|---|---|\n";
        for (const auto& b : backends_j) {
            md << "| " << b["backend"].get<std::string>() << " | " << b["implementation"].get<std::string>() << " | "
               << b["status"].get<std::string>() << " | " << (b["reason"].is_null() ? "" : b["reason"].get<std::string>())
               << " |\n";
        }
        md << "\n## Pooled medians (ms)\n\n| backend | metric | frame | input | median | p95 | n |\n|---|---|---|---|---|---|---|\n";
        for (const auto& g : groups_j) {
            md << "| " << g["backend"].get<std::string>() << " | " << g["metric"].get<std::string>() << " | "
               << g["frame_id"].get<std::string>() << " | " << g["input_size"][0].dump() << "x" << g["input_size"][1].dump()
               << " | " << fmt(g["pooled_within_runs"]["median_ms"].get<double>()) << " | "
               << fmt(g["pooled_within_runs"]["p95_ms"].get<double>()) << " | "
               << g["pooled_within_runs"]["count"].dump() << " |\n";
        }
        md << "\n## Limitations\n\n"
           << "- Timings are from one machine and one session; see environment.json.\n"
           << "- " << summary["workload_note"].get<std::string>() << "\n"
           << "- Speedup ratios compare like-for-like medians only; null entries were not measured.\n"
           << "- Unavailable backends were not measured; no zero or placeholder values were substituted.\n";
        if (zero_samples > 0) md << "- " << zero_samples << " samples were at or below timer resolution.\n";
        io::write_text_file(out("analysis.md"), md.str(), true);
    }

    std::error_code rm_ec;
    fs::remove_all(fs::path(opt.output_dir) / "scratch", rm_ec);
    std::cerr << "benchmark: wrote " << samples.size() << " samples to " << opt.output_dir << " (status "
              << summary["status"].get<std::string>() << ")\n";
    if (any_failed) {
        bool correctness = false;
        for (const auto& r : runners) correctness = correctness || r.status == "failed_correctness";
        return correctness ? exit_code::kCorrectnessFailure : exit_code::kProcessingFailure;
    }
    if (!any_measured) return exit_code::kBackendUnavailable;
    return exit_code::kSuccess;
}

}  // namespace framepipe::bench
