// config.cpp - 期权/模拟参数文件解析
// 格式：每行 "key = value"，'#' 起为注释；兼容 CRLF 与 UTF-8 BOM（Windows 编辑）
#include "config.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string lower(const std::string& s) {
    std::string r = s;
    for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

// 读取文件为 key -> (value, 行号)（跳过空行/纯注释行，剥离 # 注释与 \r）
std::map<std::string, std::pair<std::string, int>> read_kv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw ConfigError(0, "cannot open file: " + path);
    std::map<std::string, std::pair<std::string, int>> kv;
    std::string raw;
    int line_no = 0;
    bool first_line = true;
    while (std::getline(in, raw)) {
        ++line_no;
        if (first_line) {
            first_line = false;
            // 跳过 UTF-8 BOM
            if (raw.size() >= 3 && static_cast<unsigned char>(raw[0]) == 0xEF &&
                static_cast<unsigned char>(raw[1]) == 0xBB && static_cast<unsigned char>(raw[2]) == 0xBF)
                raw = raw.substr(3);
        }
        if (!raw.empty() && raw.back() == '\r') raw.pop_back(); // CRLF
        std::string line = trim(raw);
        if (line.empty() || line[0] == '#') continue;
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = trim(line.substr(0, hash));
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos)
            throw ConfigError(line_no, path + ": expected 'key = value', got: " + line);
        std::string key = lower(trim(line.substr(0, eq)));
        std::string val = trim(line.substr(eq + 1));
        // 允许字符串值带双引号（PDF 示例格式 rng = "curand"），数字同样兼容
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);
        if (key.empty() || val.empty()) throw ConfigError(line_no, path + ": empty key or value");
        if (kv.count(key)) throw ConfigError(line_no, path + ": duplicate key: " + key);
        kv[key] = {val, line_no};
    }
    return kv;
}

// 带行号记录的解析辅助
class Parser {
public:
    explicit Parser(const std::string& path, std::map<std::string, std::pair<std::string, int>> kv)
        : kv_(std::move(kv)), path_(path) {}

    double required(const char* key) {
        auto it = kv_.find(key);
        if (it == kv_.end()) fail(0, std::string("missing required key: ") + key);
        return number(key, it->second.first, it->second.second);
    }

    double optional(const char* key, double def) {
        auto it = kv_.find(key);
        return it == kv_.end() ? def : number(key, it->second.first, it->second.second);
    }

    std::string required_string(const char* key) {
        auto it = kv_.find(key);
        if (it == kv_.end()) fail(0, std::string("missing required key: ") + key);
        return it->second.first;
    }

    std::string optional_string(const char* key, const std::string& def) {
        auto it = kv_.find(key);
        return it == kv_.end() ? def : it->second.first;
    }

    void reject_unknown(std::initializer_list<const char*> known) {
        for (const auto& [k, v] : kv_) {
            bool ok = false;
            for (const char* kk : known)
                if (k == kk) { ok = true; break; }
            if (!ok) fail(v.second, "unknown key: " + k);
        }
    }

private:
    double number(const char* key, const std::string& val, int line) {
        size_t pos = 0;
        double x = 0.0;
        try {
            x = std::stod(val, &pos);
        } catch (...) {
            fail(line, std::string("invalid number for ") + key + ": " + val);
        }
        if (pos != val.size()) fail(line, std::string("invalid number for ") + key + ": " + val);
        return x;
    }

    [[noreturn]] void fail(int line, const std::string& msg) {
        throw ConfigError(line, path_ + ": " + msg);
    }

    std::map<std::string, std::pair<std::string, int>> kv_;
    std::string path_;
};

} // namespace

OptionSpec parse_option_file(const std::string& path) {
    Parser p(path, read_kv(path));
    p.reject_unknown({"option_type", "spot", "strike", "risk_free_rate", "volatility",
                      "maturity", "barrier"});

    std::string t = lower(p.required_string("option_type"));
    OptionSpec os;
    if (t == "european_call") os.type = OptionType::EuropeanCall;
    else if (t == "european_put") os.type = OptionType::EuropeanPut;
    else if (t == "asian_call") os.type = OptionType::AsianCall;
    else if (t == "barrier_call") os.type = OptionType::BarrierCallUpOut;
    else throw ConfigError(0, path + ": unknown option_type: " + t);

    os.spot = p.required("spot");
    os.strike = p.required("strike");
    os.risk_free_rate = p.required("risk_free_rate");
    os.volatility = p.required("volatility");
    os.maturity = p.required("maturity");

    if (!(os.spot > 0)) throw ConfigError(0, path + ": spot must be > 0");
    if (!(os.strike > 0)) throw ConfigError(0, path + ": strike must be > 0");
    if (!(os.volatility > 0)) throw ConfigError(0, path + ": volatility must be > 0");
    if (!(os.maturity > 0)) throw ConfigError(0, path + ": maturity must be > 0");
    if (!std::isfinite(os.risk_free_rate))
        throw ConfigError(0, path + ": risk_free_rate must be finite");

    if (os.type == OptionType::BarrierCallUpOut) {
        os.barrier = p.required("barrier");
        if (!(os.barrier > os.spot))
            throw ConfigError(0, path + ": barrier must be > spot for barrier_call");
    } else {
        os.barrier = std::nan(""); // 非障碍期权不使用
    }
    return os;
}

SimSpec parse_sim_file(const std::string& path) {
    Parser p(path, read_kv(path));
    p.reject_unknown({"num_paths", "num_steps", "seed", "rng", "variance_reduction"});

    SimSpec ss;
    ss.num_paths = static_cast<long long>(p.optional("num_paths", 10000000));
    ss.num_steps = static_cast<int>(p.optional("num_steps", 256));
    ss.seed = static_cast<unsigned long long>(p.optional("seed", 1234));
    if (ss.num_paths <= 0) throw ConfigError(0, path + ": num_paths must be > 0");
    if (ss.num_steps <= 0) throw ConfigError(0, path + ": num_steps must be > 0");

    std::string rng = lower(p.optional_string("rng", "curand"));
    if (rng == "curand") ss.rng = RngType::Curand;
    else if (rng == "custom") ss.rng = RngType::Custom;
    else throw ConfigError(0, path + ": unknown rng: " + rng);

    std::string vr = lower(p.optional_string("variance_reduction", "none"));
    if (vr == "none") ss.vr = VarReduction::None;
    else if (vr == "antithetic") ss.vr = VarReduction::Antithetic;
    else if (vr == "control_variate") ss.vr = VarReduction::ControlVariate;
    else throw ConfigError(0, path + ": unknown variance_reduction: " + vr);

    if (ss.vr == VarReduction::Antithetic && ss.num_paths % 2 != 0)
        throw ConfigError(0, path + ": num_paths must be even for antithetic (pairs)");
    return ss;
}
