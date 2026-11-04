#pragma once

#include <string>
#include <vector>

namespace framepipe::app {

// Entry point used by main(); returns a process exit code. All diagnostics go
// to stderr; stdout carries only requested machine-readable output.
int run_cli(const std::vector<std::string>& args);

}  // namespace framepipe::app
