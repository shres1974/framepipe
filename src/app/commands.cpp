// CLI commands: doctor, process, batch, validate, benchmark (spec Section 14).

#include "app/commands.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "app/batch.hpp"
#include "app/cli_args.hpp"
#include "benchmark/environment.hpp"
#include "benchmark/harness.hpp"
#include "core/sha256.hpp"
#include "framepipe/build_info.hpp"
#include "framepipe/errors.hpp"
#include "framepipe/pipeline.hpp"
#include "io/image_io.hpp"
#include "io/json_util.hpp"
#include "io/manifest.hpp"
#include "validation/compare.hpp"

namespace framepipe::app {

namespace fs = std::filesystem;
using ojson = nlohmann::ordered_json;
using clk = std::chrono::steady_clock;

namespace {

const char* kTopHelp =
    "FramePipe " FRAMEPIPE_VERSION " - reproducible CPU/CUDA image preprocessing\n"
    "\n"
    "Usage: framepipe <command> [options]\n"
    "\n"
    "Commands:\n"
    "  doctor      Describe build capabilities and usable runtime backends\n"
    "  process     Process one image (grayscale -> resize -> blur -> edges)\n"
    "  batch       Process an ordered JSON Lines image manifest\n"
    "  validate    Compare a candidate backend against cpu-ref or oracle fixtures\n"
    "  benchmark   Measure declared timing scopes and save raw samples\n"
    "\n"
    "Run 'framepipe <command> --help' for command options.\n"
    "Exit codes: 0 success, 2 invalid arguments/configuration, 3 I/O failure,\n"
    "            4 unavailable backend, 5 processing/resource failure, 6 correctness failure.\n";

// ---------------------------------------------------------------- config options
const std::vector<OptionSpec> kConfigOptions = {
    {"config", true, "JSON configuration file (schema_version 1)", "PATH"},
    {"backend", true, "auto | cpu-ref | cpu-omp | cuda (overrides config)", "NAME"},
    {"device-index", true, "CUDA device index (overrides config)", "N"},
    {"output-stage", true, "resize | blur | edges (overrides config)", "STAGE"},
    {"width", true, "resize width (overrides config)", "N"},
    {"height", true, "resize height (overrides config)", "N"},
    {"radius", true, "Gaussian radius 0..15 (overrides config)", "N"},
    {"sigma", true, "Gaussian sigma 0.1..100 (overrides config)", "X"},
    {"threads", true, "cpu-omp thread count (overrides config)", "N"},
};

std::vector<OptionSpec> with_config(std::vector<OptionSpec> v) {
    v.insert(v.end(), kConfigOptions.begin(), kConfigOptions.end());
    v.push_back({"help", false, "Show this help"});
    return v;
}

double parse_double(const std::string& name, const std::string& v) {
    try {
        std::size_t pos = 0;
        const double d = std::stod(v, &pos);
        if (pos != v.size()) throw std::invalid_argument("trailing");
        return d;
    } catch (const std::exception&) {
        throw Error(ErrorCategory::InvalidArgument, "--" + name + " expects a number, got '" + v + "'");
    }
}

// Precedence: built-in defaults < JSON file < explicit CLI overrides.
std::string cli_overrides_json(const ParsedArgs& a) {
    ojson j;
    j["schema_version"] = 1;
    if (a.has("backend")) j["backend"] = *a.get("backend");
    if (a.has("device-index")) j["device_index"] = a.get_int("device-index", 0);
    if (a.has("output-stage")) j["output_stage"] = *a.get("output-stage");
    if (a.has("width")) j["resize"]["width"] = a.get_int("width", 0);
    if (a.has("height")) j["resize"]["height"] = a.get_int("height", 0);
    if (a.has("radius")) j["gaussian"]["radius"] = a.get_int("radius", 0);
    if (a.has("sigma")) {
        const double s = parse_double("sigma", *a.get("sigma"));
        if (!std::isfinite(s)) throw Error(ErrorCategory::InvalidConfiguration, "gaussian.sigma must be finite");
        j["gaussian"]["sigma"] = s;
    }
    if (a.has("threads")) j["cpu"]["threads"] = a.get_int("threads", 1);
    return j.dump();
}

PipelineConfig resolve_config(const ParsedArgs& a) {
    PipelineConfig c;
    if (a.has("config")) c = apply_config_json(io::read_text_file(*a.get("config")), c);
    return apply_config_json(cli_overrides_json(a), c);
}

ojson resolved_json(const Pipeline& p) {
    const auto& r = p.configuration();
    ojson j;
    j["resolved_config"] = ojson::parse(config_to_json(r.config));
    j["selected_backend"] = backend_name(r.selected_backend);
    j["implementation"] = r.implementation;
    j["selection_reason"] = r.selection_reason;
    j["math_config_hash"] = r.math_config_hash;
    return j;
}

void ensure_parent_dir(const std::string& path) {
    const fs::path parent = fs::path(path).parent_path();
    if (parent.empty()) return;
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) throw Error(ErrorCategory::IoFailure, "cannot create output directory: " + parent.string(), {"", "", "", ec.message()});
}

void check_output_path(const std::string& out, const std::vector<std::string>& inputs, bool overwrite) {
    std::error_code ec;
    for (const auto& in : inputs) {
        if (fs::exists(out, ec) && fs::equivalent(in, out, ec)) {
            throw Error(ErrorCategory::InvalidArgument, "output path resolves to the same file as the input: " + out);
        }
        if (fs::weakly_canonical(in, ec) == fs::weakly_canonical(out, ec)) {
            throw Error(ErrorCategory::InvalidArgument, "output path resolves to the same file as the input: " + out);
        }
    }
    if (!overwrite && fs::exists(out, ec)) {
        throw Error(ErrorCategory::IoFailure, "refusing to overwrite existing file (use --overwrite): " + out);
    }
}

// "-" writes JSON to stdout (and nothing else goes to stdout).
void emit_report(const std::string& dest, const ojson& j, bool overwrite) {
    io::require_finite_numbers(j, "$");
    if (dest == "-") {
        std::cout << j.dump(2) << std::endl;
        return;
    }
    ensure_parent_dir(dest);
    io::write_json_file(dest, j, overwrite);
}

bool want_help(const std::vector<std::string>& args) {
    for (const auto& a : args) {
        if (a == "--help" || a == "-h") return true;
    }
    return false;
}

// ---------------------------------------------------------------- doctor
int cmd_doctor(const std::vector<std::string>& args) {
    const std::vector<OptionSpec> specs{{"json", false, "Write machine-readable JSON to stdout"},
                                        {"device-index", true, "CUDA device index to probe (default 0)", "N"},
                                        {"help", false, "Show this help"}};
    if (want_help(args)) {
        std::cout << format_help("framepipe doctor [--json]",
                                 "Reports build capabilities and runtime backend usability independently.\n"
                                 "A CPU-only environment is a successful result, not an error.",
                                 specs);
        return 0;
    }
    const auto a = parse_options(args, specs);
    const int dev = a.get_int("device-index", 0);
    ojson j;
    j["framepipe_version"] = FRAMEPIPE_VERSION;
    j["build"] = bench::build_capabilities_json();
    ojson backends = ojson::array();
    for (Backend b : {Backend::CpuReference, Backend::CpuParallel, Backend::Cuda}) {
        const auto av = query_backend(b, dev);
        backends.push_back({{"backend", backend_name(b)}, {"built", av.built}, {"usable", av.usable}, {"reason", av.reason}});
    }
    j["backends"] = backends;
    {
        PipelineConfig c;
        c.backend = Backend::Auto;
        c.device_index = dev;
        Pipeline p(c);
        j["auto_selection"] = {{"selected", backend_name(p.selected_backend())},
                               {"reason", p.configuration().selection_reason}};
    }
    j["environment"] = bench::collect_environment(dev);
    if (a.has("json")) {
        std::cout << j.dump(2) << std::endl;
        return 0;
    }
    std::cerr << "FramePipe " << FRAMEPIPE_VERSION << " (" << build_info::kBuildType << ", "
              << build_info::kCxxCompilerId << " " << build_info::kCxxCompilerVersion << ")\n";
    std::cerr << "OpenCV runtime: " << j["environment"]["dependencies"]["opencv_runtime"].get<std::string>() << "\n";
    for (const auto& b : backends) {
        std::cerr << "  " << b["backend"].get<std::string>() << ": built=" << (b["built"].get<bool>() ? "yes" : "no")
                  << " usable=" << (b["usable"].get<bool>() ? "yes" : "no") << " (" << b["reason"].get<std::string>()
                  << ")\n";
    }
    std::cerr << "auto selects: " << j["auto_selection"]["selected"].get<std::string>() << " - "
              << j["auto_selection"]["reason"].get<std::string>() << "\n";
    return 0;
}

// ---------------------------------------------------------------- process
int cmd_process(const std::vector<std::string>& args) {
    const auto specs = with_config({{"input", true, "Input PNG/JPEG (8-bit gray or RGB)", "PATH"},
                                    {"output", true, "Output 8-bit PNG", "PATH"},
                                    {"report", true, "Write a JSON report ('-' for stdout)", "PATH"},
                                    {"overwrite", false, "Allow overwriting existing output/report files"}});
    if (want_help(args)) {
        std::cout << format_help("framepipe process --input IN --output OUT.png [options]",
                                 "Processes one image with the resolved configuration and writes an 8-bit PNG.\n"
                                 "EXIF orientation is ignored; pixels are processed in stored order.",
                                 specs);
        return 0;
    }
    const auto a = parse_options(args, specs);
    if (!a.has("input") || !a.has("output")) throw Error(ErrorCategory::InvalidArgument, "--input and --output are required");
    const std::string in = *a.get("input"), outp = *a.get("output");
    const bool overwrite = a.has("overwrite");
    const PipelineConfig cfg = resolve_config(a);
    check_output_path(outp, {in}, overwrite);
    if (a.has("report") && *a.get("report") != "-") check_output_path(*a.get("report"), {in, outp}, overwrite);

    Pipeline p(cfg);
    const auto t0 = clk::now();
    const std::string bytes = io::read_file_bytes(in);
    const auto t_read = clk::now();
    auto dec = io::decode_image(bytes, in, cfg.resources.max_dimension);
    const auto t_decode = clk::now();
    p.prepare(spec_of(dec.image.view()));
    const auto t_prep = clk::now();
    HostFloatImage out(cfg.resize.width, cfg.resize.height);
    p.process_into(dec.image.view(), out.view());
    const auto t_proc = clk::now();
    const auto q = io::quantize_image(out.cview());
    ensure_parent_dir(outp);
    io::write_png_gray(outp, q);
    const auto t_end = clk::now();
    auto d = [](clk::time_point x, clk::time_point y) { return std::chrono::duration<double, std::milli>(y - x).count(); };
    std::cerr << "process: " << in << " -> " << outp << " (" << shape_string(spec_of(dec.image.view())) << " -> "
              << shape_string(cfg.resize.width, cfg.resize.height) << ", stage " << to_string(cfg.output_stage)
              << ", backend " << backend_name(p.selected_backend()) << ")\n";

    if (a.has("report")) {
        ojson r;
        r["schema_version"] = 1;
        r["command"] = "process";
        r["status"] = "completed";
        r["input"] = {{"path", in},
                      {"sha256", sha256_hex(bytes)},
                      {"width", dec.image.width()},
                      {"height", dec.image.height()},
                      {"format", to_string(dec.image.format())},
                      {"decoded_from", dec.source_format},
                      {"orientation_policy", "stored pixel order; EXIF orientation ignored"}};
        r["output"] = {{"path", outp},
                       {"width", cfg.resize.width},
                       {"height", cfg.resize.height},
                       {"stage", to_string(cfg.output_stage)},
                       {"quantization", "uint8(floor(clamp(v,0,1)*255+0.5))"}};
        auto rj = resolved_json(p);
        for (auto it = rj.begin(); it != rj.end(); ++it) r[it.key()] = it.value();
        r["timing"] = {
            {"note", "single cold invocation; diagnostic breakdown, not a benchmark. Use 'framepipe benchmark'."},
            {"application_wall_ms", d(t0, t_end)},
            {"read_ms", d(t0, t_read)},
            {"decode_canonicalize_ms", d(t_read, t_decode)},
            {"prepare_ms", d(t_decode, t_prep)},
            {"process_ms", d(t_prep, t_proc)},
            {"quantize_encode_write_ms", d(t_proc, t_end)},
            {"process_state", "cold (first frame in this process)"}};
        const auto w = p.workspace();
        r["workspace"] = {{"host_logical_bytes", w.host_logical_bytes},
                          {"host_allocated_bytes", w.host_allocated_bytes},
                          {"device_logical_bytes", w.device_logical_bytes},
                          {"device_allocated_bytes", w.device_allocated_bytes},
                          {"pinned_host_bytes", w.pinned_host_bytes}};
        emit_report(*a.get("report"), r, overwrite);
    }
    return 0;
}

// ---------------------------------------------------------------- batch
int cmd_batch(const std::vector<std::string>& args) {
    const auto specs = with_config(
        {{"manifest", true, "JSON Lines manifest: {\"frame_id\",\"path\",\"source_timestamp_ns\"?}", "PATH"},
         {"output-dir", true, "Directory for <frame_id>.png outputs and outputs.jsonl", "DIR"},
         {"report", true, "Batch report path (default OUTPUT_DIR/batch_report.json; '-' for stdout)", "PATH"},
         {"continue-on-error", false, "Record failed frames and continue (default: fail fast)"},
         {"queue-capacity", true, "Bounded decode queue capacity (default 2)", "N"},
         {"overwrite", false, "Allow overwriting existing outputs"}});
    if (want_help(args)) {
        std::cout << format_help("framepipe batch --manifest M.jsonl --output-dir DIR [options]",
                                 "Processes every manifest frame in manifest order with blocking backpressure.\n"
                                 "No frames are dropped. Output files are named <frame_id>.png.",
                                 specs);
        return 0;
    }
    const auto a = parse_options(args, specs);
    if (!a.has("manifest") || !a.has("output-dir")) {
        throw Error(ErrorCategory::InvalidArgument, "--manifest and --output-dir are required");
    }
    BatchOptions bo;
    bo.config = resolve_config(a);
    bo.output_dir = *a.get("output-dir");
    bo.overwrite = a.has("overwrite");
    bo.continue_on_error = a.has("continue-on-error");
    const int qc = a.get_int("queue-capacity", 2);
    if (qc < 1 || qc > 64) throw Error(ErrorCategory::InvalidArgument, "--queue-capacity must be in [1, 64]");
    bo.queue_capacity = static_cast<std::size_t>(qc);
    const io::Manifest m = io::load_manifest(*a.get("manifest"), true);
    const std::string report = a.get_or("report", (fs::path(bo.output_dir) / "batch_report.json").string());
    const std::string mapping = (fs::path(bo.output_dir) / "outputs.jsonl").string();
    std::error_code ec;
    if (!bo.overwrite && fs::exists(mapping, ec)) {
        throw Error(ErrorCategory::IoFailure, "refusing to overwrite existing file (use --overwrite): " + mapping);
    }
    if (report != "-" && !bo.overwrite && fs::exists(report, ec)) {
        throw Error(ErrorCategory::IoFailure, "refusing to overwrite existing file (use --overwrite): " + report);
    }

    const BatchResult res = run_batch(m, bo);

    std::ostringstream lines;
    ojson frames = ojson::array();
    for (const auto& f : res.frames) {
        ojson j;
        j["sequence"] = f.sequence;
        j["frame_id"] = f.frame_id;
        j["input_path"] = f.input_path;
        j["output_path"] = f.output_path.empty() ? ojson(nullptr) : ojson(f.output_path);
        j["source_timestamp_ns"] = f.source_timestamp_ns ? ojson(*f.source_timestamp_ns) : ojson(nullptr);
        j["status"] = f.status;
        j["error"] = f.error.empty() ? ojson(nullptr) : ojson(f.error);
        if (f.status != "not_processed") {
            j["accepted_ms"] = f.accepted_ms;
            j["completed_ms"] = f.completed_ms;
        }
        lines << j.dump() << "\n";
        frames.push_back(j);
    }
    io::write_text_file(mapping, lines.str(), true);

    ojson r;
    r["schema_version"] = 1;
    r["command"] = "batch";
    r["status"] = res.failed == 0 && res.not_processed == 0 ? "completed" : "failed";
    r["policy"] = {{"backpressure", "blocking (no drops)"},
                   {"order", "manifest order"},
                   {"on_error", bo.continue_on_error ? "continue_and_record" : "fail_fast"},
                   {"queue_capacity", bo.queue_capacity}};
    r["manifest"] = {{"path", m.path}, {"sha256", m.sha256}, {"frames", m.entries.size()}};
    r["counts"] = {{"submitted", res.submitted},
                   {"completed", res.completed},
                   {"failed", res.failed},
                   {"not_processed", res.not_processed},
                   {"dropped", 0}};
    r["batch_wall_ms"] = res.batch_wall_ms;
    r["throughput_completed_fps"] =
        res.batch_wall_ms > 0 ? ojson(static_cast<double>(res.completed) / (res.batch_wall_ms / 1000.0)) : ojson(nullptr);
    r["throughput_definition"] = "completed frames / batch wall seconds (single cold run, includes startup of reader)";
    r["reprepare_count"] = res.reprepare_count;
    r["queue_high_water"] = res.queue_high_water;
    r["selected_backend"] = res.selected_backend;
    r["implementation"] = res.implementation;
    r["selection_reason"] = res.selection_reason;
    r["resolved_config"] = ojson::parse(config_to_json(bo.config));
    r["math_config_hash"] = math_config_hash(bo.config);
    r["frames"] = frames;
    emit_report(report, r, true);
    std::cerr << "batch: " << res.completed << " completed, " << res.failed << " failed, " << res.not_processed
              << " not processed (" << res.selected_backend << ")\n";
    if (res.failed > 0) {
        for (const auto& f : res.frames) {
            if (f.status == "failed") std::cerr << "  " << f.frame_id << ": " << f.error << "\n";
        }
        return exit_code::kProcessingFailure;
    }
    return 0;
}

// ---------------------------------------------------------------- validate
// Oracle fixture format: <dir>/<stage>.f32 = ASCII header "FPF32 <w> <h>\n"
// followed by w*h little-endian float32 values in row-major order.
std::optional<HostFloatImage> load_f32(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return std::nullopt;
    std::string magic;
    int w = 0, h = 0;
    in >> magic >> w >> h;
    in.get();
    if (magic != "FPF32" || w <= 0 || h <= 0) throw Error(ErrorCategory::IoFailure, "malformed oracle file: " + p.string());
    HostFloatImage im(w, h);
    for (int y = 0; y < h; ++y) {
        in.read(reinterpret_cast<char*>(&im.at(0, y)), static_cast<std::streamsize>(static_cast<std::size_t>(w) * 4));
    }
    if (!in) throw Error(ErrorCategory::IoFailure, "truncated oracle file: " + p.string());
    return im;
}

int cmd_validate(const std::vector<std::string>& args) {
    const auto specs = with_config(
        {{"input", true, "Input PNG/JPEG", "PATH"},
         {"candidate", true, "Candidate backend: cpu-ref | cpu-omp | cuda (default cpu-ref)", "NAME"},
         {"reference", true, "cpu-ref (default) or oracle:DIR with <stage>.f32 files from scripts/oracle.py", "REF"},
         {"report", true, "Write JSON report ('-' for stdout)", "PATH"},
         {"overwrite", false, "Allow overwriting the report"}});
    if (want_help(args)) {
        std::cout << format_help(
            "framepipe validate --input IN [--candidate cuda] [--reference cpu-ref|oracle:DIR] [options]",
            "Compares every stage's FP32 output (before quantization) under the per-pixel rule\n"
            "abs(c-r) <= atol + rtol*abs(r) with fixed per-stage tolerances. Exit 6 on mismatch.\n"
            "cpu-ref vs cpu-ref is only an interface smoke check, not independent validation.",
            specs);
        return 0;
    }
    const auto a = parse_options(args, specs);
    if (!a.has("input")) throw Error(ErrorCategory::InvalidArgument, "--input is required");
    if (a.has("backend")) throw Error(ErrorCategory::InvalidArgument, "use --candidate instead of --backend for validate");
    PipelineConfig cfg = resolve_config(a);
    const std::string cand_name = a.get_or("candidate", "cpu-ref");
    const std::string ref_name = a.get_or("reference", "cpu-ref");
    const Backend cand_backend = parse_backend(cand_name);
    if (cand_backend == Backend::Auto) throw Error(ErrorCategory::InvalidArgument, "--candidate must name a concrete backend");
    const bool oracle = ref_name.rfind("oracle:", 0) == 0;
    if (!oracle && ref_name != "cpu-ref") {
        throw Error(ErrorCategory::InvalidArgument, "--reference must be cpu-ref or oracle:DIR");
    }
    if (a.has("report") && *a.get("report") != "-") check_output_path(*a.get("report"), {*a.get("input")}, a.has("overwrite"));

    const auto dec = io::read_image(*a.get("input"), cfg.resources.max_dimension);
    const auto spec = spec_of(dec.image.view());

    PipelineConfig cc = cfg;
    cc.backend = cand_backend;
    Pipeline cand(cc);
    cand.prepare(spec);
    const StageCapture cs = cand.process_capture_stages(dec.image.view());
    // Also run the normal (non-capture) path so its final output is checked too.
    HostFloatImage normal(cfg.resize.width, cfg.resize.height);
    cand.process_into(dec.image.view(), normal.view());

    StageCapture rs;
    std::string ref_label;
    if (oracle) {
        const fs::path dir = ref_name.substr(7);
        for (Stage s : stages_for(cfg.output_stage, true)) {
            auto im = load_f32(dir / (std::string(to_string(s)) + ".f32"));
            if (im) rs.images[static_cast<std::size_t>(s)] = std::move(*im);
        }
        ref_label = "independent NumPy oracle (" + dir.string() + ")";
    } else {
        PipelineConfig rc = cfg;
        rc.backend = Backend::CpuReference;
        Pipeline ref(rc);
        ref.prepare(spec);
        rs = ref.process_capture_stages(dec.image.view());
        ref_label = "cpu-ref";
    }
    const bool self = !oracle && cand_backend == Backend::CpuReference;

    ojson stages = ojson::array();
    bool all = true;
    int compared = 0;
    for (Stage s : stages_for(cfg.output_stage, true)) {
        const auto* r = rs.get(s);
        const auto* c = cs.get(s);
        ojson sj;
        sj["stage"] = to_string(s);
        if (!r || !c) {
            sj["status"] = "not_executed";
            sj["reason"] = !r ? "reference stage not available" : "candidate stage not available";
            stages.push_back(sj);
            continue;
        }
        const auto cr = compare_float_images(r->cview(), c->cview(), tolerance_for(s), s == Stage::Edges);
        auto cj = ojson::parse(compare_result_json(cr));
        sj.update(cj);
        stages.push_back(sj);
        all = all && cr.passed;
        ++compared;
    }
    // Normal-path final output must match the capture-path final output.
    const Stage final_stage = cfg.output_stage == OutputStage::Edges  ? Stage::Edges
                              : cfg.output_stage == OutputStage::Blur ? Stage::BlurVertical
                                                                      : Stage::Resized;
    const auto np = compare_float_images(cs.get(final_stage)->cview(), normal.cview(), tolerance_for(final_stage),
                                         cfg.output_stage == OutputStage::Edges);
    all = all && np.passed;
    // Supplemental 8-bit parity (after float parity).
    ojson u8;
    if (const auto* rf = rs.get(final_stage)) {
        const auto qa = io::quantize_image(rf->cview());
        const auto qb = io::quantize_image(normal.cview());
        int maxd = 0;
        for (int y = 0; y < qa.height(); ++y) {
            for (int x = 0; x < qa.width(); ++x) maxd = std::max(maxd, std::abs(int(qa.row(y)[x]) - int(qb.row(y)[x])));
        }
        u8 = {{"max_abs_level_difference", maxd}, {"tolerance", 1}, {"status", maxd <= 1 ? "passed" : "failed"}};
        all = all && maxd <= 1;
    } else {
        u8 = {{"status", "not_executed"}};
    }
    if (compared == 0) all = false;

    ojson r;
    r["schema_version"] = 1;
    r["command"] = "validate";
    r["status"] = all ? "passed" : "failed";
    r["validation_kind"] = self ? "interface_smoke_check_self_comparison"
                                : (oracle ? "candidate_vs_independent_oracle" : "candidate_vs_cpu_ref");
    if (self) {
        r["note"] = "cpu-ref compared with itself only checks the CLI/report path. Independent CPU correctness "
                    "evidence comes from the NumPy oracle suite (ctest -L oracle) and hand fixtures.";
    }
    r["input"] = {{"path", *a.get("input")}, {"width", spec.width}, {"height", spec.height}, {"format", to_string(spec.format)}};
    r["reference"] = ref_label;
    r["candidate"] = backend_name(cand.selected_backend());
    r["candidate_implementation"] = cand.configuration().implementation;
    r["resolved_config"] = ojson::parse(config_to_json(cfg));
    r["math_config_hash"] = math_config_hash(cfg);
    r["stages"] = stages;
    r["normal_path_vs_capture_path"] = ojson::parse(compare_result_json(np));
    r["quantized_u8_parity"] = u8;
    if (a.has("report")) emit_report(*a.get("report"), r, a.has("overwrite"));
    std::cerr << "validate: " << (all ? "PASSED" : "FAILED") << " (" << backend_name(cand.selected_backend()) << " vs "
              << ref_label << ", " << compared << " stages compared)\n";
    if (!all) {
        for (const auto& s : stages) {
            if (s.value("status", "") != "passed") {
                std::cerr << "  stage " << s["stage"].get<std::string>() << ": " << s.value("message", s.value("reason", "")) << "\n";
            }
        }
    }
    return all ? 0 : exit_code::kCorrectnessFailure;
}

// ---------------------------------------------------------------- benchmark
int cmd_benchmark(const std::vector<std::string>& args, const std::string& cmdline) {
    std::vector<OptionSpec> specs{
        {"manifest", true, "JSON Lines manifest of inputs", "PATH"},
        {"config", true, "JSON configuration file", "PATH"},
        {"output-stage", true, "resize | blur | edges (overrides config)", "STAGE"},
        {"width", true, "resize width (overrides config)", "N"},
        {"height", true, "resize height (overrides config)", "N"},
        {"radius", true, "Gaussian radius (overrides config)", "N"},
        {"sigma", true, "Gaussian sigma (overrides config)", "X"},
        {"device-index", true, "CUDA device index", "N"},
        {"backends", true, "Comma list: cpu-ref,cpu-omp,cuda,opencv-cpu", "LIST"},
        {"scopes", true, "Comma list: host_processing,device_pipeline,application,startup,batch", "LIST"},
        {"workload", true, "repeat (default) or ring", "NAME"},
        {"warmup", true, "Warm-up iterations per frame (default 20)", "N"},
        {"iterations", true, "Measured iterations per frame (default 200)", "N"},
        {"repetitions", true, "Repeated runs (default 5)", "N"},
        {"threads", true, "cpu-omp threads (default: config cpu.threads)", "N"},
        {"opencv-threads", true, "OpenCV baseline threads (default 1)", "N"},
        {"output-dir", true, "New result directory", "DIR"},
        {"overwrite", false, "Allow reusing an existing result directory"},
        {"help", false, "Show this help"}};
    if (want_help(args)) {
        std::cout << format_help(
            "framepipe benchmark --manifest M.jsonl --backends cpu-ref,cuda --scopes host_processing --output-dir DIR",
            "Writes run.json, environment.json, resolved_config.json, inputs.jsonl, validation.json,\n"
            "samples.jsonl, summary.json, summary.csv, analysis.md. Each backend is validated against\n"
            "cpu-ref before measurement. Unavailable backends are recorded, never synthesized.",
            specs);
        return 0;
    }
    const auto a = parse_options(args, specs);
    if (!a.has("manifest") || !a.has("output-dir")) {
        throw Error(ErrorCategory::InvalidArgument, "--manifest and --output-dir are required");
    }
    bench::BenchmarkOptions o;
    o.manifest_path = *a.get("manifest");
    o.config_path = a.get_or("config", "");
    ojson ov;
    ov["schema_version"] = 1;
    if (a.has("output-stage")) ov["output_stage"] = *a.get("output-stage");
    if (a.has("width")) ov["resize"]["width"] = a.get_int("width", 0);
    if (a.has("height")) ov["resize"]["height"] = a.get_int("height", 0);
    if (a.has("radius")) ov["gaussian"]["radius"] = a.get_int("radius", 0);
    if (a.has("sigma")) ov["gaussian"]["sigma"] = parse_double("sigma", *a.get("sigma"));
    if (a.has("device-index")) ov["device_index"] = a.get_int("device-index", 0);
    if (a.has("threads")) ov["cpu"]["threads"] = a.get_int("threads", 1);
    o.config_overrides_json = ov.dump();
    o.backends = split_csv(a.get_or("backends", "cpu-ref"));
    o.scopes = split_csv(a.get_or("scopes", "host_processing"));
    o.workload = a.get_or("workload", "repeat");
    o.warmup = a.get_int("warmup", 20);
    o.iterations = a.get_int("iterations", 200);
    o.repetitions = a.get_int("repetitions", 5);
    o.output_dir = *a.get("output-dir");
    o.overwrite = a.has("overwrite");
    o.omp_threads = a.get_int("threads", 0);
    o.opencv_threads = a.get_int("opencv-threads", 1);
    if (o.opencv_threads < 1) throw Error(ErrorCategory::InvalidArgument, "--opencv-threads must be >= 1");
    o.command_line = cmdline;
    return bench::run_benchmark(o);
}

}  // namespace

int run_cli(const std::vector<std::string>& args) {
    if (args.empty() || args[0] == "--help" || args[0] == "-h" || args[0] == "help") {
        (args.empty() ? std::cerr : std::cout) << kTopHelp;
        return args.empty() ? exit_code::kInvalidArguments : 0;
    }
    if (args[0] == "--version") {
        std::cout << "framepipe " << FRAMEPIPE_VERSION << "\n";
        return 0;
    }
    const std::string cmd = args[0];
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    std::string cmdline = "framepipe";
    for (const auto& s : args) cmdline += " " + s;
    try {
        if (cmd == "doctor") return cmd_doctor(rest);
        if (cmd == "process") return cmd_process(rest);
        if (cmd == "batch") return cmd_batch(rest);
        if (cmd == "validate") return cmd_validate(rest);
        if (cmd == "benchmark") return cmd_benchmark(rest, cmdline);
        std::cerr << "framepipe: unknown command '" << cmd << "'\n\n" << kTopHelp;
        return exit_code::kInvalidArguments;
    } catch (const Error& e) {
        std::cerr << "framepipe " << cmd << ": error: " << e.what() << "\n";
        return exit_code_for(e.category());
    } catch (const std::bad_alloc&) {
        std::cerr << "framepipe " << cmd << ": error: out of host memory\n";
        return exit_code::kProcessingFailure;
    } catch (const std::exception& e) {
        std::cerr << "framepipe " << cmd << ": internal error: " << e.what() << "\n";
        return exit_code::kProcessingFailure;
    }
}

}  // namespace framepipe::app
