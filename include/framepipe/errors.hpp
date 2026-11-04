#pragma once
// FramePipe error model: typed exceptions carrying a category and context.
// See docs/decisions/0002-error-mechanism.md.

#include <stdexcept>
#include <string>

namespace framepipe {

enum class ErrorCategory {
    InvalidArgument,       // malformed CLI arguments or API misuse
    InvalidConfiguration,  // configuration fails schema or range validation
    InvalidImage,          // image view violates the geometry/stride/extent contract
    UnsupportedFormat,     // decoded input is a format P0 does not accept
    BackendUnavailable,    // requested backend not built or not usable at runtime
    AllocationFailure,     // host/device allocation failed or exceeds a budget
    DeviceFailure,         // GPU runtime error during processing
    IoFailure,             // file read/write/decode/encode failure
    CorrectnessFailure,    // validation found a mismatch outside tolerance
    Internal               // violated internal invariant (a bug)
};

const char* to_string(ErrorCategory c) noexcept;

struct ErrorContext {
    std::string operation;  // e.g. "prepare", "gaussian_horizontal"
    std::string backend;    // e.g. "cpu-ref", "cuda"
    std::string shape;      // e.g. "1920x1080 rgb8"
    std::string detail;     // underlying runtime message, if any
};

class Error : public std::runtime_error {
public:
    Error(ErrorCategory category, const std::string& message, ErrorContext context = {});

    ErrorCategory category() const noexcept { return category_; }
    const std::string& message() const noexcept { return message_; }
    const ErrorContext& context() const noexcept { return context_; }

private:
    ErrorCategory category_;
    std::string message_;
    ErrorContext context_;
};

// CLI exit codes (docs/decisions/0004-exit-codes.md).
namespace exit_code {
constexpr int kSuccess = 0;
constexpr int kInvalidArguments = 2;
constexpr int kIoFailure = 3;
constexpr int kBackendUnavailable = 4;
constexpr int kProcessingFailure = 5;
constexpr int kCorrectnessFailure = 6;
}  // namespace exit_code

int exit_code_for(ErrorCategory c) noexcept;

}  // namespace framepipe
