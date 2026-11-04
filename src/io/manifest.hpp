#pragma once
// JSON Lines manifest (spec Section 15). One record per line:
//   {"frame_id": "f0001", "path": "frames/0001.png", "source_timestamp_ns": 123}
// Manifest order is authoritative. Paths are relative to the manifest
// directory unless absolute.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace framepipe::io {

struct ManifestEntry {
    std::string frame_id;
    std::string path;           // as written in the manifest
    std::string resolved_path;  // absolute or manifest-relative resolution
    std::optional<std::int64_t> source_timestamp_ns;
    int line_number = 0;
};

struct Manifest {
    std::string path;
    std::string sha256;
    std::vector<ManifestEntry> entries;
};

// Parses and validates the whole manifest before processing. Rejects
// malformed records, unknown fields, duplicate IDs, unsafe IDs, and
// nonexistent inputs (when check_files). Throws Error(InvalidConfiguration)
// or Error(IoFailure).
Manifest load_manifest(const std::string& path, bool check_files = true);

// A frame_id is safe for use in output filenames: 1-128 chars of
// [A-Za-z0-9._-], not starting with '.', so it cannot escape a directory.
bool is_safe_frame_id(const std::string& id) noexcept;

}  // namespace framepipe::io
