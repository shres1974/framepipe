#include "framepipe/config.hpp"

#include <cmath>
#include <cstdint>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

#include "core/sha256.hpp"
#include "framepipe/errors.hpp"
#include "framepipe/image_view.hpp"

namespace framepipe {

using ojson = nlohmann::ordered_json;

const char* to_string(Backend b) noexcept {
    switch (b) {
        case Backend::Auto: return "auto";
        case Backend::CpuReference: return "cpu-ref";
        case Backend::CpuParallel: return "cpu-omp";
        case Backend::Cuda: return "cuda";
    }
    return "unknown";
}

const char* to_string(OutputStage s) noexcept {
    switch (s) {
        case OutputStage::Resize: return "resize";
        case OutputStage::Blur: return "blur";
        case OutputStage::Edges: return "edges";
    }
    return "unknown";
}

const char* to_string(CudaImplementation i) noexcept {
    switch (i) {
        case CudaImplementation::Basic: return "basic";
    }
    return "unknown";
}

namespace {
[[noreturn]] void config_error(const std::string& msg) { throw Error(ErrorCategory::InvalidConfiguration, msg); }
}  // namespace

Backend parse_backend(const std::string& s) {
    // "hip" is deliberately not accepted: it does not exist (spec Section 8).
    if (s == "auto") return Backend::Auto;
    if (s == "cpu-ref") return Backend::CpuReference;
    if (s == "cpu-omp") return Backend::CpuParallel;
    if (s == "cuda") return Backend::Cuda;
    config_error("unsupported backend '" + s + "' (accepted: auto, cpu-ref, cpu-omp, cuda)");
}

OutputStage parse_output_stage(const std::string& s) {
    if (s == "resize") return OutputStage::Resize;
    if (s == "blur") return OutputStage::Blur;
    if (s == "edges") return OutputStage::Edges;
    config_error("unsupported output_stage '" + s + "' (accepted: resize, blur, edges)");
}

CudaImplementation parse_cuda_implementation(const std::string& s) {
    if (s == "basic") return CudaImplementation::Basic;
    config_error("unsupported cuda.implementation '" + s + "' (accepted: basic)");
}

void validate_config(const PipelineConfig& c) {
    if (c.schema_version != 1) config_error("schema_version must be 1, got " + std::to_string(c.schema_version));
    if (c.device_index < 0) config_error("device_index must be >= 0");
    if (c.resources.max_dimension < 1 || c.resources.max_dimension > kHardMaxDimension) {
        config_error("resources.max_dimension must be in [1, " + std::to_string(kHardMaxDimension) + "], got " +
                     std::to_string(c.resources.max_dimension));
    }
    if (c.resize.width < 1 || c.resize.height < 1) {
        config_error("resize.width and resize.height must be positive, got " +
                     shape_string(c.resize.width, c.resize.height));
    }
    if (c.resize.width > c.resources.max_dimension || c.resize.height > c.resources.max_dimension) {
        config_error("resize dimensions " + shape_string(c.resize.width, c.resize.height) +
                     " exceed resources.max_dimension " + std::to_string(c.resources.max_dimension));
    }
    if (c.resize.mode != "bilinear_half_pixel") {
        config_error("unsupported resize.mode '" + c.resize.mode + "' (accepted: bilinear_half_pixel)");
    }
    if (c.resize.aspect_policy != "stretch") {
        config_error("unsupported resize.aspect_policy '" + c.resize.aspect_policy + "' (accepted: stretch)");
    }
    if (c.gaussian.radius < 0 || c.gaussian.radius > kMaxGaussianRadius) {
        config_error("gaussian.radius must be in [0, " + std::to_string(kMaxGaussianRadius) + "], got " +
                     std::to_string(c.gaussian.radius));
    }
    if (!std::isfinite(c.gaussian.sigma) || c.gaussian.sigma < kMinSigma || c.gaussian.sigma > kMaxSigma) {
        std::ostringstream os;
        os << "gaussian.sigma must be finite and in [0.1, 100.0], got " << c.gaussian.sigma;
        config_error(os.str());
    }
    if (c.border != "replicate") config_error("unsupported border '" + c.border + "' (accepted: replicate)");
    if (c.cpu.threads < 1 || c.cpu.threads > kMaxCpuThreads) {
        config_error("cpu.threads must be in [1, " + std::to_string(kMaxCpuThreads) + "]");
    }
    if (c.resources.max_in_flight_frames < 1 || c.resources.max_in_flight_frames > kMaxInFlightFrames) {
        config_error("resources.max_in_flight_frames must be in [1, " + std::to_string(kMaxInFlightFrames) + "]");
    }
    if (c.resources.max_device_bytes == 0) config_error("resources.max_device_bytes must be positive");
}

namespace {

void check_keys(const ojson& obj, const std::set<std::string>& allowed, const std::string& where) {
    if (!obj.is_object()) config_error(where.empty() ? "configuration must be a JSON object" : where + " must be an object");
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if (allowed.count(it.key()) == 0) {
            config_error("unknown configuration field '" + (where.empty() ? "" : where + ".") + it.key() + "'");
        }
    }
}

int get_int(const ojson& v, const std::string& name) {
    if (!v.is_number_integer()) config_error(name + " must be an integer");
    const auto x = v.get<std::int64_t>();
    if (x < INT32_MIN || x > INT32_MAX) config_error(name + " is out of range");
    return static_cast<int>(x);
}

std::uint64_t get_u64(const ojson& v, const std::string& name) {
    if (!v.is_number_unsigned()) config_error(name + " must be a non-negative integer");
    return v.get<std::uint64_t>();
}

double get_number(const ojson& v, const std::string& name) {
    if (!v.is_number() ) config_error(name + " must be a number");
    return v.get<double>();
}

std::string get_string(const ojson& v, const std::string& name) {
    if (!v.is_string()) config_error(name + " must be a string");
    return v.get<std::string>();
}

}  // namespace

PipelineConfig apply_config_json(const std::string& json_text, const PipelineConfig& base) {
    ojson j;
    try {
        j = ojson::parse(json_text);
    } catch (const nlohmann::json::parse_error& e) {
        config_error(std::string("configuration is not valid JSON: ") + e.what());
    }
    PipelineConfig c = base;
    check_keys(j, {"schema_version", "backend", "device_index", "output_stage", "resize", "gaussian", "border", "cpu",
                   "cuda", "resources"},
               "");
    if (!j.contains("schema_version")) config_error("configuration must declare schema_version");
    c.schema_version = get_int(j["schema_version"], "schema_version");
    if (c.schema_version != 1) config_error("unsupported schema_version " + std::to_string(c.schema_version));
    if (j.contains("backend")) c.backend = parse_backend(get_string(j["backend"], "backend"));
    if (j.contains("device_index")) c.device_index = get_int(j["device_index"], "device_index");
    if (j.contains("output_stage")) c.output_stage = parse_output_stage(get_string(j["output_stage"], "output_stage"));
    if (j.contains("border")) c.border = get_string(j["border"], "border");
    if (j.contains("resize")) {
        const auto& r = j["resize"];
        check_keys(r, {"width", "height", "mode", "aspect_policy"}, "resize");
        if (r.contains("width")) c.resize.width = get_int(r["width"], "resize.width");
        if (r.contains("height")) c.resize.height = get_int(r["height"], "resize.height");
        if (r.contains("mode")) c.resize.mode = get_string(r["mode"], "resize.mode");
        if (r.contains("aspect_policy")) c.resize.aspect_policy = get_string(r["aspect_policy"], "resize.aspect_policy");
    }
    if (j.contains("gaussian")) {
        const auto& g = j["gaussian"];
        check_keys(g, {"radius", "sigma"}, "gaussian");
        if (g.contains("radius")) c.gaussian.radius = get_int(g["radius"], "gaussian.radius");
        if (g.contains("sigma")) c.gaussian.sigma = get_number(g["sigma"], "gaussian.sigma");
    }
    if (j.contains("cpu")) {
        const auto& p = j["cpu"];
        check_keys(p, {"threads"}, "cpu");
        if (p.contains("threads")) c.cpu.threads = get_int(p["threads"], "cpu.threads");
    }
    if (j.contains("cuda")) {
        const auto& p = j["cuda"];
        check_keys(p, {"implementation"}, "cuda");
        if (p.contains("implementation")) {
            c.cuda.implementation = parse_cuda_implementation(get_string(p["implementation"], "cuda.implementation"));
        }
    }
    if (j.contains("resources")) {
        const auto& p = j["resources"];
        check_keys(p, {"max_dimension", "max_device_bytes", "max_pinned_host_bytes", "max_in_flight_frames"},
                   "resources");
        if (p.contains("max_dimension")) c.resources.max_dimension = get_int(p["max_dimension"], "resources.max_dimension");
        if (p.contains("max_device_bytes")) c.resources.max_device_bytes = get_u64(p["max_device_bytes"], "resources.max_device_bytes");
        if (p.contains("max_pinned_host_bytes")) {
            c.resources.max_pinned_host_bytes = get_u64(p["max_pinned_host_bytes"], "resources.max_pinned_host_bytes");
        }
        if (p.contains("max_in_flight_frames")) {
            c.resources.max_in_flight_frames = get_int(p["max_in_flight_frames"], "resources.max_in_flight_frames");
        }
    }
    validate_config(c);
    return c;
}

namespace {
ojson to_ojson(const PipelineConfig& c) {
    ojson j;
    j["schema_version"] = c.schema_version;
    j["backend"] = to_string(c.backend);
    j["device_index"] = c.device_index;
    j["output_stage"] = to_string(c.output_stage);
    j["resize"] = {{"width", c.resize.width},
                   {"height", c.resize.height},
                   {"mode", c.resize.mode},
                   {"aspect_policy", c.resize.aspect_policy}};
    j["gaussian"] = {{"radius", c.gaussian.radius}, {"sigma", c.gaussian.sigma}};
    j["border"] = c.border;
    j["cpu"] = {{"threads", c.cpu.threads}};
    j["cuda"] = {{"implementation", to_string(c.cuda.implementation)}};
    j["resources"] = {{"max_dimension", c.resources.max_dimension},
                      {"max_device_bytes", c.resources.max_device_bytes},
                      {"max_pinned_host_bytes", c.resources.max_pinned_host_bytes},
                      {"max_in_flight_frames", c.resources.max_in_flight_frames}};
    return j;
}
}  // namespace

std::string config_to_json(const PipelineConfig& c, int indent) { return to_ojson(c).dump(indent); }

std::string math_config_hash(const PipelineConfig& c) {
    // Canonical, versioned encoding. sigma uses 17 significant digits so the
    // exact double value is represented.
    std::ostringstream os;
    os.precision(17);
    os << "framepipe-math-v1;stage=" << to_string(c.output_stage) << ";resize=" << c.resize.width << "x"
       << c.resize.height << ";mode=" << c.resize.mode << ";aspect=" << c.resize.aspect_policy
       << ";radius=" << c.gaussian.radius << ";sigma=" << c.gaussian.sigma << ";border=" << c.border;
    return sha256_hex(os.str());
}

}  // namespace framepipe
