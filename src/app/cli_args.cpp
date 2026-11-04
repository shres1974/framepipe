#include "app/cli_args.hpp"

#include <cstdint>
#include <sstream>

#include "framepipe/errors.hpp"

namespace framepipe::app {

std::optional<std::string> ParsedArgs::get(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) return std::nullopt;
    return it->second;
}

std::string ParsedArgs::get_or(const std::string& name, const std::string& def) const {
    auto v = get(name);
    return v ? *v : def;
}

int ParsedArgs::get_int(const std::string& name, int def) const {
    auto v = get(name);
    if (!v) return def;
    try {
        std::size_t pos = 0;
        const long long x = std::stoll(*v, &pos);
        if (pos != v->size() || x < INT32_MIN || x > INT32_MAX) throw std::invalid_argument("range");
        return static_cast<int>(x);
    } catch (const std::exception&) {
        throw Error(ErrorCategory::InvalidArgument, "--" + name + " expects an integer, got '" + *v + "'");
    }
}

ParsedArgs parse_options(const std::vector<std::string>& args, const std::vector<OptionSpec>& specs) {
    ParsedArgs out;
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string a = args[i];
        if (a.rfind("--", 0) != 0 || a.size() < 3) {
            throw Error(ErrorCategory::InvalidArgument, "unexpected argument '" + a + "' (options start with --)");
        }
        a = a.substr(2);
        std::optional<std::string> inline_value;
        const auto eq = a.find('=');
        if (eq != std::string::npos) {
            inline_value = a.substr(eq + 1);
            a = a.substr(0, eq);
        }
        const OptionSpec* spec = nullptr;
        for (const auto& s : specs) {
            if (s.name == a) spec = &s;
        }
        if (!spec) throw Error(ErrorCategory::InvalidArgument, "unknown option '--" + a + "' (see --help)");
        if (out.has(a)) throw Error(ErrorCategory::InvalidArgument, "option '--" + a + "' given more than once");
        if (spec->takes_value) {
            if (inline_value) {
                out.values_[a] = *inline_value;
            } else {
                if (i + 1 >= args.size()) throw Error(ErrorCategory::InvalidArgument, "option '--" + a + "' requires a value");
                out.values_[a] = args[++i];
            }
        } else {
            if (inline_value) throw Error(ErrorCategory::InvalidArgument, "option '--" + a + "' does not take a value");
            out.values_[a] = "";
        }
    }
    return out;
}

std::string format_help(const std::string& usage, const std::string& description, const std::vector<OptionSpec>& specs) {
    std::ostringstream os;
    os << "Usage: " << usage << "\n\n" << description << "\n\nOptions:\n";
    for (const auto& s : specs) {
        std::string left = "  --" + s.name + (s.takes_value ? " " + s.value_name : "");
        if (left.size() < 30) left.resize(30, ' ');
        else left += "\n" + std::string(30, ' ');
        os << left << s.help << "\n";
    }
    return os.str();
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> v;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) v.push_back(cur);
            cur.clear();
        } else if (c != ' ') {
            cur += c;
        }
    }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

}  // namespace framepipe::app
