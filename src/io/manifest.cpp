#include "io/manifest.hpp"

#include <filesystem>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

#include "core/sha256.hpp"
#include "framepipe/errors.hpp"
#include "io/json_util.hpp"

namespace framepipe::io {

namespace fs = std::filesystem;

bool is_safe_frame_id(const std::string& id) noexcept {
    if (id.empty() || id.size() > 128 || id[0] == '.') return false;
    for (char c : id) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

Manifest load_manifest(const std::string& path, bool check_files) {
    const std::string text = read_text_file(path);
    Manifest m;
    m.path = path;
    m.sha256 = sha256_hex(text);
    const fs::path base = fs::absolute(fs::path(path)).parent_path();
    std::istringstream in(text);
    std::string line;
    int ln = 0;
    std::set<std::string> ids;
    auto bad = [&](const std::string& msg) {
        throw Error(ErrorCategory::InvalidConfiguration, "manifest " + path + " line " + std::to_string(ln) + ": " + msg);
    };
    while (std::getline(in, line)) {
        ++ln;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(line);
        } catch (const nlohmann::json::parse_error&) {
            bad("malformed JSON record");
        }
        if (!j.is_object()) bad("record must be a JSON object");
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() != "frame_id" && it.key() != "path" && it.key() != "source_timestamp_ns") {
                bad("unknown field '" + it.key() + "'");
            }
        }
        if (!j.contains("frame_id") || !j["frame_id"].is_string()) bad("frame_id must be a string");
        if (!j.contains("path") || !j["path"].is_string()) bad("path must be a string");
        ManifestEntry e;
        e.line_number = ln;
        e.frame_id = j["frame_id"].get<std::string>();
        e.path = j["path"].get<std::string>();
        if (!is_safe_frame_id(e.frame_id)) {
            bad("frame_id '" + e.frame_id + "' must be 1-128 characters of [A-Za-z0-9._-] and not start with '.'");
        }
        if (!ids.insert(e.frame_id).second) bad("duplicate frame_id '" + e.frame_id + "'");
        if (e.path.empty()) bad("path must not be empty");
        if (j.contains("source_timestamp_ns")) {
            if (!j["source_timestamp_ns"].is_number_integer()) bad("source_timestamp_ns must be an integer");
            e.source_timestamp_ns = j["source_timestamp_ns"].get<std::int64_t>();
        }
        const fs::path p(e.path);
        e.resolved_path = (p.is_absolute() ? p : base / p).lexically_normal().string();
        if (check_files) {
            std::error_code ec;
            if (!fs::is_regular_file(e.resolved_path, ec)) bad("input does not exist or is not a file: " + e.resolved_path);
        }
        m.entries.push_back(std::move(e));
    }
    if (m.entries.empty()) throw Error(ErrorCategory::InvalidConfiguration, "manifest " + path + " has no records");
    return m;
}

}  // namespace framepipe::io
