#include <string>
#include <vector>

#include "app/commands.hpp"

int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    return framepipe::app::run_cli(args);
}
