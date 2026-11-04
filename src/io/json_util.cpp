#include "io/json_util.hpp"

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

#include "framepipe/errors.hpp"

namespace framepipe::io {

namespace fs = std::filesystem;

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error(ErrorCategory::IoFailure, "cannot open file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    if (in.bad()) throw Error(ErrorCategory::IoFailure, "read failed: " + path);
    return ss.str();
}

void write_text_file(const std::string& path, const std::string& text, bool overwrite) {
    std::error_code ec;
    if (!overwrite && fs::exists(path, ec)) {
        throw Error(ErrorCategory::IoFailure, "refusing to overwrite existing file (use --overwrite): " + path);
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw Error(ErrorCategory::IoFailure, "cannot open file for writing: " + path);
    out << text;
    out.close();
    if (!out) throw Error(ErrorCategory::IoFailure, "write failed: " + path);
}

void require_finite_numbers(const ojson& j, const std::string& where) {
    if (j.is_number_float()) {
        if (!std::isfinite(j.get<double>())) {
            throw Error(ErrorCategory::Internal, "non-finite number in JSON output at " + where);
        }
    } else if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it) require_finite_numbers(it.value(), where + "." + it.key());
    } else if (j.is_array()) {
        for (std::size_t i = 0; i < j.size(); ++i) require_finite_numbers(j[i], where + "[" + std::to_string(i) + "]");
    }
}

void write_json_file(const std::string& path, const ojson& j, bool overwrite) {
    require_finite_numbers(j, "$");
    write_text_file(path, j.dump(2) + "\n", overwrite);
}

std::string utc_timestamp_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string make_run_id() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
    std::random_device rd;
    std::ostringstream os;
    os << buf << "-" << std::hex << (rd() & 0xffffu) << "-" << ::getpid();
    return os.str();
}

}  // namespace framepipe::io
