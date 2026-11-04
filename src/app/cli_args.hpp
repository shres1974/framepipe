#pragma once
// Small documented argument parser (docs/decisions/0003-cli-parser.md).
// Grammar: <command> [--flag] [--option value]...  Options may also be given
// as --option=value. Unknown options are errors.

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace framepipe::app {

struct OptionSpec {
    std::string name;  // without leading dashes
    bool takes_value;
    std::string help;
    std::string value_name = "VALUE";
};

class ParsedArgs {
public:
    bool has(const std::string& name) const { return values_.count(name) > 0; }
    std::optional<std::string> get(const std::string& name) const;
    std::string get_or(const std::string& name, const std::string& def) const;
    int get_int(const std::string& name, int def) const;  // throws InvalidArgument
    std::map<std::string, std::string> values_;
};

// Throws Error(InvalidArgument) with a specific message.
ParsedArgs parse_options(const std::vector<std::string>& args, const std::vector<OptionSpec>& specs);

std::string format_help(const std::string& usage, const std::string& description, const std::vector<OptionSpec>& specs);

std::vector<std::string> split_csv(const std::string& s);

}  // namespace framepipe::app
