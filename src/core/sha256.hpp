#pragma once
// Minimal SHA-256 (FIPS 180-4) for configuration, manifest, and file hashes.

#include <cstddef>
#include <cstdint>
#include <string>

namespace framepipe {

class Sha256 {
public:
    Sha256();
    void update(const void* data, std::size_t len);
    void update(const std::string& s) { update(s.data(), s.size()); }
    std::string hex_digest();  // finalizes; object must not be reused afterwards

private:
    void block(const std::uint8_t* p);
    std::uint32_t h_[8];
    std::uint8_t buf_[64];
    std::size_t buf_len_ = 0;
    std::uint64_t total_len_ = 0;
};

std::string sha256_hex(const std::string& data);
// Throws Error(IoFailure) if the file cannot be read.
std::string sha256_file_hex(const std::string& path);

}  // namespace framepipe
