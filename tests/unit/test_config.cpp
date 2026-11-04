#include "framepipe/config.hpp"
#include "framepipe/errors.hpp"
#include "test_util.hpp"

using namespace framepipe;

namespace {
void expect_config_error(const std::string& json, const std::string& needle) {
    try {
        apply_config_json(json);
        FAIL() << "expected InvalidConfiguration for " << json;
    } catch (const Error& e) {
        EXPECT_EQ(e.category(), ErrorCategory::InvalidConfiguration);
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}
}  // namespace

TEST(Config, DefaultsMatchSpec) {
    PipelineConfig c;
    EXPECT_EQ(c.schema_version, 1);
    EXPECT_EQ(c.backend, Backend::Auto);
    EXPECT_EQ(c.device_index, 0);
    EXPECT_EQ(c.output_stage, OutputStage::Edges);
    EXPECT_EQ(c.resize.width, 1280);
    EXPECT_EQ(c.resize.height, 720);
    EXPECT_EQ(c.resize.mode, "bilinear_half_pixel");
    EXPECT_EQ(c.resize.aspect_policy, "stretch");
    EXPECT_EQ(c.gaussian.radius, 3);
    EXPECT_DOUBLE_EQ(c.gaussian.sigma, 1.2);
    EXPECT_EQ(c.border, "replicate");
    EXPECT_EQ(c.cpu.threads, 1);
    EXPECT_EQ(c.cuda.implementation, CudaImplementation::Basic);
    EXPECT_EQ(c.resources.max_dimension, 16384);
    EXPECT_EQ(c.resources.max_device_bytes, 1073741824ULL);
    EXPECT_EQ(c.resources.max_pinned_host_bytes, 268435456ULL);
    EXPECT_EQ(c.resources.max_in_flight_frames, 1);
    EXPECT_NO_THROW(validate_config(c));
}

TEST(Config, RoundTripResolved) {
    PipelineConfig c;
    c.gaussian.sigma = 0.30000000000000004;
    c.output_stage = OutputStage::Blur;
    const auto back = apply_config_json(config_to_json(c));
    EXPECT_EQ(config_to_json(back), config_to_json(c));
    EXPECT_EQ(math_config_hash(back), math_config_hash(c));
}

TEST(Config, PartialOverridesBase) {
    PipelineConfig base;
    base.resize.width = 640;
    const auto c = apply_config_json(R"({"schema_version":1,"gaussian":{"radius":5}})", base);
    EXPECT_EQ(c.resize.width, 640);
    EXPECT_EQ(c.gaussian.radius, 5);
    EXPECT_DOUBLE_EQ(c.gaussian.sigma, 1.2);
}

TEST(Config, UnknownFieldsRejected) {
    expect_config_error(R"({"schema_version":1,"bakend":"cpu-ref"})", "bakend");
    expect_config_error(R"({"schema_version":1,"resize":{"widht":3}})", "resize.widht");
    expect_config_error(R"({"schema_version":1,"gaussian":{"radius":3,"sgima":1}})", "gaussian.sgima");
}

TEST(Config, BadEnumsRejected) {
    expect_config_error(R"({"schema_version":1,"backend":"hip"})", "hip");
    expect_config_error(R"({"schema_version":1,"output_stage":"canny"})", "canny");
    expect_config_error(R"({"schema_version":1,"border":"reflect"})", "reflect");
    expect_config_error(R"({"schema_version":1,"resize":{"mode":"nearest"}})", "nearest");
    expect_config_error(R"({"schema_version":1,"resize":{"aspect_policy":"letterbox"}})", "letterbox");
    expect_config_error(R"({"schema_version":1,"cuda":{"implementation":"turbo"}})", "turbo");
}

TEST(Config, RangesEnforced) {
    expect_config_error(R"({"schema_version":1,"gaussian":{"radius":16}})", "radius");
    expect_config_error(R"({"schema_version":1,"gaussian":{"radius":-1}})", "radius");
    expect_config_error(R"({"schema_version":1,"gaussian":{"sigma":0.05}})", "sigma");
    expect_config_error(R"({"schema_version":1,"gaussian":{"sigma":100.5}})", "sigma");
    expect_config_error(R"({"schema_version":1,"resize":{"width":0}})", "positive");
    expect_config_error(R"({"schema_version":1,"resize":{"width":20000}})", "max_dimension");
    expect_config_error(R"({"schema_version":1,"resources":{"max_dimension":20000}})", "max_dimension");
    expect_config_error(R"({"schema_version":1,"resources":{"max_dimension":100}})", "max_dimension");  // < resize 1280
    expect_config_error(R"({"schema_version":1,"cpu":{"threads":0}})", "threads");
    expect_config_error(R"({"schema_version":1,"device_index":-1})", "device_index");
}

TEST(Config, TypesEnforced) {
    expect_config_error(R"({"schema_version":1,"gaussian":{"radius":3.5}})", "integer");
    expect_config_error(R"({"schema_version":1,"gaussian":{"sigma":"1.2"}})", "number");
    expect_config_error(R"({"schema_version":1,"backend":3})", "string");
    expect_config_error(R"({"schema_version":1,"resources":{"max_device_bytes":-5}})", "non-negative");
}

TEST(Config, SchemaVersionRequired) {
    expect_config_error(R"({"backend":"cpu-ref"})", "schema_version");
    expect_config_error(R"({"schema_version":2})", "schema_version");
    expect_config_error(R"([1,2])", "object");
    expect_config_error(R"({not json)", "JSON");
}

TEST(Config, NaNSigmaRejectedProgrammatically) {
    PipelineConfig c;
    c.gaussian.sigma = std::nan("");
    EXPECT_THROW(validate_config(c), Error);
    c.gaussian.sigma = std::numeric_limits<double>::infinity();
    EXPECT_THROW(validate_config(c), Error);
}

TEST(Config, HashCoversMathParametersOnly) {
    PipelineConfig a, b;
    b.backend = Backend::CpuReference;
    b.cpu.threads = 8;
    b.resources.max_device_bytes = 1;
    EXPECT_EQ(math_config_hash(a), math_config_hash(b));
    b.gaussian.sigma = 1.3;
    EXPECT_NE(math_config_hash(a), math_config_hash(b));
    PipelineConfig c;
    c.output_stage = OutputStage::Blur;
    EXPECT_NE(math_config_hash(a), math_config_hash(c));
}

TEST(Config, ExampleFilesParse) {
    // The checked-in example must equal the documented defaults.
    const std::string ex = R"({
      "schema_version": 1, "backend": "auto", "device_index": 0, "output_stage": "edges",
      "resize": {"width": 1280, "height": 720, "mode": "bilinear_half_pixel", "aspect_policy": "stretch"},
      "gaussian": {"radius": 3, "sigma": 1.2}, "border": "replicate", "cpu": {"threads": 1},
      "cuda": {"implementation": "basic"},
      "resources": {"max_dimension": 16384, "max_device_bytes": 1073741824,
                    "max_pinned_host_bytes": 268435456, "max_in_flight_frames": 1}})";
    EXPECT_EQ(config_to_json(apply_config_json(ex)), config_to_json(PipelineConfig{}));
}
