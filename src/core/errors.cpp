#include "framepipe/errors.hpp"

namespace framepipe {

const char* to_string(ErrorCategory c) noexcept {
    switch (c) {
        case ErrorCategory::InvalidArgument: return "invalid_argument";
        case ErrorCategory::InvalidConfiguration: return "invalid_configuration";
        case ErrorCategory::InvalidImage: return "invalid_image";
        case ErrorCategory::UnsupportedFormat: return "unsupported_format";
        case ErrorCategory::BackendUnavailable: return "backend_unavailable";
        case ErrorCategory::AllocationFailure: return "allocation_failure";
        case ErrorCategory::DeviceFailure: return "device_failure";
        case ErrorCategory::IoFailure: return "io_failure";
        case ErrorCategory::CorrectnessFailure: return "correctness_failure";
        case ErrorCategory::Internal: return "internal_error";
    }
    return "unknown";
}

static std::string compose(ErrorCategory category, const std::string& message, const ErrorContext& ctx) {
    std::string s = std::string(to_string(category)) + ": " + message;
    std::string extra;
    auto add = [&](const char* k, const std::string& v) {
        if (v.empty()) return;
        if (!extra.empty()) extra += ", ";
        extra += k;
        extra += "=";
        extra += v;
    };
    add("operation", ctx.operation);
    add("backend", ctx.backend);
    add("shape", ctx.shape);
    add("detail", ctx.detail);
    if (!extra.empty()) s += " [" + extra + "]";
    return s;
}

Error::Error(ErrorCategory category, const std::string& message, ErrorContext context)
    : std::runtime_error(compose(category, message, context)),
      category_(category),
      message_(message),
      context_(std::move(context)) {}

int exit_code_for(ErrorCategory c) noexcept {
    switch (c) {
        case ErrorCategory::InvalidArgument:
        case ErrorCategory::InvalidConfiguration:
            return exit_code::kInvalidArguments;
        case ErrorCategory::IoFailure:
        case ErrorCategory::UnsupportedFormat:
            return exit_code::kIoFailure;
        case ErrorCategory::BackendUnavailable:
            return exit_code::kBackendUnavailable;
        case ErrorCategory::InvalidImage:
        case ErrorCategory::AllocationFailure:
        case ErrorCategory::DeviceFailure:
        case ErrorCategory::Internal:
            return exit_code::kProcessingFailure;
        case ErrorCategory::CorrectnessFailure:
            return exit_code::kCorrectnessFailure;
    }
    return exit_code::kProcessingFailure;
}

}  // namespace framepipe
