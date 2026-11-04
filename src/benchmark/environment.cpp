#include "benchmark/environment.hpp"

#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <thread>

#include <opencv2/core.hpp>

#include "framepipe/build_info.hpp"
#include "framepipe/pipeline.hpp"

namespace framepipe::bench {

using ojson = nlohmann::ordered_json;

namespace {

std::string trim(std::string s) {
    const auto b = s.find_first_not_of(" \t\r\n\"");
    const auto e = s.find_last_not_of(" \t\r\n\"");
    if (b == std::string::npos) return "";
    return s.substr(b, e - b + 1);
}

std::string read_key(const std::string& path, const std::string& key, char sep) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto p = line.find(sep);
        if (p == std::string::npos) continue;
        if (trim(line.substr(0, p)) == key) return trim(line.substr(p + 1));
    }
    return "";
}

// Runs a fixed command (no user input is interpolated) and returns stdout.
std::string run_fixed(const std::string& cmd, int* status = nullptr) {
    std::array<char, 512> buf{};
    std::string out;
    FILE* p = ::popen(cmd.c_str(), "r");
    if (!p) {
        if (status) *status = -1;
        return "";
    }
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), p)) out += buf.data();
    const int st = ::pclose(p);
    if (status) *status = st;
    return trim(out);
}

std::string shell_quote(const std::string& s) {
    std::string r = "'";
    for (char c : s) {
        if (c == '\'') r += "'\\''";
        else r += c;
    }
    return r + "'";
}

}  // namespace

GitInfo git_info() {
    GitInfo g;
#ifdef FRAMEPIPE_SOURCE_DIR
    const std::string dir = shell_quote(FRAMEPIPE_SOURCE_DIR);
    int st = 0;
    g.revision = run_fixed("git -C " + dir + " rev-parse HEAD 2>/dev/null", &st);
    if (st != 0 || g.revision.empty()) {
        g.revision.clear();
        g.note = "git revision unavailable (not a git checkout or no commits yet)";
        return g;
    }
    g.known = true;
    const std::string status = run_fixed("git -C " + dir + " status --porcelain --untracked-files=no 2>/dev/null", &st);
    g.dirty = !status.empty();
#else
    g.note = "source directory unknown at build time";
#endif
    return g;
}

ojson build_capabilities_json() {
    ojson j;
    j["framepipe_version"] = build_info::kVersion;
    j["build_type"] = build_info::kBuildType;
    j["cxx_compiler"] = std::string(build_info::kCxxCompilerId) + " " + build_info::kCxxCompilerVersion;
    j["cxx_flags"] = trim(build_info::kCxxFlags);
    j["cmake_version"] = build_info::kCmakeVersion;
    j["opencv_version_build"] = build_info::kOpenCvVersion;
    j["openmp_built"] = build_info::kOpenMpBuilt;
    j["openmp_version"] = build_info::kOpenMpBuilt ? ojson(build_info::kOpenMpVersion) : ojson(nullptr);
    j["sanitizers"] = build_info::kSanitized;
    j["cuda_built"] = build_info::kCudaBuilt;
    j["cuda_compiler_version"] = build_info::kCudaBuilt ? ojson(build_info::kCudaCompilerVersion) : ojson(nullptr);
    j["cuda_flags"] = build_info::kCudaBuilt ? ojson(trim(build_info::kCudaFlags)) : ojson(nullptr);
    j["cuda_architectures"] = build_info::kCudaBuilt ? ojson(build_info::kCudaArchitectures) : ojson(nullptr);
    j["cuda_debug_sync"] = build_info::kCudaDebugSync;
    j["fast_math"] = false;
    return j;
}

ojson collect_environment(int device_index) {
    ojson j;
    ojson unknown = ojson::object();
    j["schema_version"] = 1;

    ojson host;
    const std::string cpu = read_key("/proc/cpuinfo", "model name", ':');
    host["cpu_model"] = cpu.empty() ? ojson(nullptr) : ojson(cpu);
    if (cpu.empty()) unknown["cpu_model"] = "/proc/cpuinfo has no 'model name' entry";
    const long online = ::sysconf(_SC_NPROCESSORS_ONLN);
    host["logical_cpus"] = online > 0 ? ojson(online) : ojson(nullptr);
    host["hardware_concurrency"] = std::thread::hardware_concurrency();
    const std::string mem = read_key("/proc/meminfo", "MemTotal", ':');
    if (!mem.empty()) {
        std::istringstream ms(mem);
        long long kb = 0;
        ms >> kb;
        host["ram_bytes"] = kb * 1024LL;
    } else {
        host["ram_bytes"] = nullptr;
        unknown["ram_bytes"] = "/proc/meminfo unavailable";
    }
    struct utsname u {};
    if (::uname(&u) == 0) {
        host["kernel"] = std::string(u.sysname) + " " + u.release;
        host["machine"] = u.machine;
    }
    const std::string os = read_key("/etc/os-release", "PRETTY_NAME", '=');
    host["os"] = os.empty() ? ojson(nullptr) : ojson(os);
    if (os.empty()) unknown["os"] = "/etc/os-release unavailable";
    {
        std::ifstream g("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
        std::string v;
        if (g && std::getline(g, v)) {
            host["cpu_scaling_governor"] = trim(v);
        } else {
            host["cpu_scaling_governor"] = nullptr;
            unknown["cpu_scaling_governor"] = "cpufreq not exposed (common in VMs/WSL)";
        }
    }
    host["virtualization_hint"] = (u.release[0] && std::string(u.release).find("microsoft") != std::string::npos)
                                      ? ojson("WSL2 (Microsoft kernel)")
                                      : ojson(nullptr);
    j["host"] = host;

    j["build"] = build_capabilities_json();

    ojson deps;
    deps["opencv_runtime"] = cv::getVersionString();
    deps["opencv_threads"] = cv::getNumThreads();
    deps["nlohmann_json"] = std::to_string(NLOHMANN_JSON_VERSION_MAJOR) + "." +
                            std::to_string(NLOHMANN_JSON_VERSION_MINOR) + "." +
                            std::to_string(NLOHMANN_JSON_VERSION_PATCH);
    j["dependencies"] = deps;

    const auto cuda = query_cuda_runtime();
    ojson c;
    c["built"] = cuda.built;
    c["toolkit_version"] = cuda.toolkit_version.empty() ? ojson(nullptr) : ojson(cuda.toolkit_version);
    c["runtime_version"] = cuda.runtime_version ? ojson(*cuda.runtime_version) : ojson(nullptr);
    c["driver_version"] = cuda.driver_version ? ojson(*cuda.driver_version) : ojson(nullptr);
    c["device_count"] = cuda.device_count;
    c["selected_device_index"] = device_index;
    c["error"] = cuda.error.empty() ? ojson(nullptr) : ojson(cuda.error);
    ojson devs = ojson::array();
    for (const auto& d : cuda.devices) {
        devs.push_back({{"index", d.index},
                        {"name", d.name},
                        {"compute_capability", std::to_string(d.compute_major) + "." + std::to_string(d.compute_minor)},
                        {"total_memory_bytes", d.total_memory_bytes},
                        {"multiprocessors", d.multiprocessors}});
    }
    c["devices"] = devs;
    if (cuda.devices.empty()) unknown["gpu"] = cuda.built ? "no usable CUDA device: " + cuda.error : "CUDA backend not built";
    j["cuda"] = c;

    ojson settings;
    settings["omp_num_threads_env"] = std::getenv("OMP_NUM_THREADS") ? ojson(std::getenv("OMP_NUM_THREADS")) : ojson(nullptr);
    settings["cuda_visible_devices_env"] =
        std::getenv("CUDA_VISIBLE_DEVICES") ? ojson(std::getenv("CUDA_VISIBLE_DEVICES")) : ojson(nullptr);
    j["settings"] = settings;
    j["unknown_reasons"] = unknown;
    return j;
}

}  // namespace framepipe::bench
