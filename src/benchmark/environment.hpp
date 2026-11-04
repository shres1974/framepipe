#pragma once
// Environment and build metadata. Unknown facts are null with an explanation
// in "unknown_reasons", never guessed strings (spec Section 19).

#include <nlohmann/json.hpp>
#include <string>

namespace framepipe::bench {

nlohmann::ordered_json collect_environment(int device_index);
nlohmann::ordered_json build_capabilities_json();

struct GitInfo {
    std::string revision;  // empty if unknown
    bool dirty = false;
    bool known = false;
    std::string note;
};
GitInfo git_info();

}  // namespace framepipe::bench
