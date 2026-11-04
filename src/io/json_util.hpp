#pragma once
// JSON helpers: atomic-ish file writing, finite number checking, UTC time.

#include <string>

#include <nlohmann/json.hpp>

namespace framepipe::io {

using ojson = nlohmann::ordered_json;

std::string read_text_file(const std::string& path);
// Writes text; throws Error(IoFailure). If !overwrite and the file exists, throws.
void write_text_file(const std::string& path, const std::string& text, bool overwrite);
void write_json_file(const std::string& path, const ojson& j, bool overwrite);

// Throws Error(Internal) if any number in j is NaN or infinite.
void require_finite_numbers(const ojson& j, const std::string& where);

std::string utc_timestamp_iso8601();
std::string make_run_id();

}  // namespace framepipe::io
