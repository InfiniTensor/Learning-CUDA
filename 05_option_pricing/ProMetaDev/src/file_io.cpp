// ============================================================
// file_io.cpp - 参数文件解析 / 结果输出 (CSV + JSON) / 性能日志
// ============================================================
#include "file_io.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cctype>
#include <iostream>
#include <initializer_list>
#include <stdexcept>

// -------- 工具: 去空白 --------
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// -------- 工具: 转大写（用于键名/枚举值的宽松匹配） --------
static std::string to_upper(std::string s) {
    for (char& c : s)
        c = (char)std::toupper((unsigned char)c);
    return s;
}

// -------- 工具: 去掉值两侧的成对引号 --------
// 题目文档中的取值写作 rng = "curand" / variance_reduction = "none"，
// 引号属于文档书写习惯，不应被当作值的一部分。
static std::string strip_quotes(std::string v) {
    if (v.size() >= 2 &&
        ((v.front() == '"' && v.back() == '"') || (v.front() == '\'' && v.back() == '\''))) {
        return v.substr(1, v.size() - 2);
    }
    return v;
}

// -------- 工具: 键名别名判断 --------
// 题目文档给出的参数文件使用 option_type / risk_free_rate，
// 而本项目历史参数文件使用 type / risk_free；两者都接受，避免
// "用题目给的参数文件时键名不匹配 → 静默使用默认值 → 价格算错"。
static bool key_is(const std::string& k, std::initializer_list<const char*> names) {
    for (const char* n : names)
        if (k == n)
            return true;
    return false;
}

// -------- 工具: 收集未识别键名并在解析结束后统一告警 --------
static void warn_unknown_keys(const std::string& file, const std::vector<std::string>& unknown) {
    if (unknown.empty())
        return;
    std::cerr << "[warn] " << file << ": 未识别的参数键（已忽略）:";
    for (const auto& k : unknown)
        std::cerr << " " << k;
    std::cerr << "\n";
}

// -------- 通用 key=value 解析器 --------
static std::vector<std::pair<std::string, std::string>> read_kv(const std::string& path) {
    std::ifstream in(path);
    // 文件打不开必须报错：否则解析结果为空，调用方会静默沿用结构体默认值，
    // 用一组"看似合理"的参数出一个错误的价格。
    if (!in)
        throw std::runtime_error("无法打开参数文件: " + path);
    std::vector<std::pair<std::string, std::string>> kv;
    std::string line;
    while (std::getline(in, line)) {
        // 跳过注释和空行
        size_t h = line.find_first_not_of(" \t");
        if (h == std::string::npos || line[h] == '#')
            continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::string k = trim(line.substr(0, eq));
        std::string v = strip_quotes(trim(line.substr(eq + 1)));
        if (!k.empty())
            kv.emplace_back(k, v);
    }
    return kv;
}

// -------- 解析期权参数 --------
OptionParams parse_option_params(const std::string& path) {
    OptionParams o;
    std::vector<std::string> unknown;
    for (auto& [k, v] : read_kv(path)) {
        // 支持题目文档键名（option_type / risk_free_rate）与项目历史键名
        if (key_is(k, {"type", "option_type"})) {
            o.type = OptionParams::parse_option_type(v);
            if (o.type == OptionType::UNKNOWN)
                throw std::runtime_error("option_params: 无法识别的 option_type = " + v);
        } else if (k == "spot")
            o.spot = std::stod(v);
        else if (k == "strike")
            o.strike = std::stod(v);
        else if (key_is(k, {"risk_free", "risk_free_rate", "rate"}))
            o.risk_free = std::stod(v);
        else if (k == "volatility")
            o.volatility = std::stod(v);
        else if (k == "maturity")
            o.maturity = std::stod(v);
        else if (k == "barrier")
            o.barrier = std::stod(v);
        else if (k == "barrier_dir")
            o.barrier_dir = (to_upper(v) == "UP") ? BarrierDir::UP : BarrierDir::DOWN;
        else if (key_is(k, {"barrier_type", "barrier_t"}))
            o.barrier_t =
                (to_upper(v) == "KNOCK_IN") ? BarrierType::KNOCK_IN : BarrierType::KNOCK_OUT;
        else
            unknown.push_back(k);
    }
    warn_unknown_keys(path, unknown);

    // -------- 参数合法性校验 --------
    // 不做校验的话，矛盾配置（例如 down 障碍却设在现价之上）会让 MC 在 t=0
    // 立即敲出、解析式同样返回 0，最终打印 "price=0 |err|=0"，看起来像完美结果。
    auto bad = [&](const std::string& msg) {
        throw std::runtime_error("option_params 校验失败: " + msg);
    };
    if (o.spot <= 0.0)
        bad("spot 必须为正");
    if (o.strike <= 0.0)
        bad("strike 必须为正");
    if (o.volatility <= 0.0)
        bad("volatility 必须为正");
    if (o.maturity <= 0.0)
        bad("maturity 必须为正");
    if (o.type == OptionType::BARRIER_CALL || o.type == OptionType::BARRIER_PUT) {
        if (o.barrier <= 0.0)
            bad("障碍期权的 barrier 必须为正");
        if (o.barrier_dir == BarrierDir::UP && o.barrier <= o.spot)
            bad("barrier_dir=UP 要求 barrier > spot（否则会在 t=0 立即触发）");
        if (o.barrier_dir == BarrierDir::DOWN && o.barrier >= o.spot)
            bad("barrier_dir=DOWN 要求 barrier < spot（否则会在 t=0 立即触发）");
    }
    return o;
}

// -------- 解析模拟参数 --------
SimParams parse_sim_params(const std::string& path) {
    SimParams s;
    std::vector<std::string> unknown;
    for (auto& [k, v] : read_kv(path)) {
        if (k == "num_paths")
            s.num_paths = std::stoll(v);
        else if (k == "num_steps")
            s.num_steps = std::stoi(v);
        else if (k == "seed")
            s.seed = (unsigned int)std::stoul(v);
        else if (k == "block_size")
            s.block_size = std::stoi(v);
        else if (k == "rng") {
            // CUSTOM_XORWOW 已移除, 始终使用 curand Philox4_32_10
            s.rng = RNGType::CURAND_DEFAULT;
        } else if (k == "variance_reduction") {
            // 大小写宽松：题目文档用小写 (none/antithetic/control_variate)
            const std::string u = to_upper(v);
            if (u == "ANTITHETIC")
                s.variance_reduction = VarianceReduction::ANTITHETIC;
            else if (u == "CONTROL_VARIATE")
                s.variance_reduction = VarianceReduction::CONTROL_VARIATE;
            else if (u == "NONE" || u.empty())
                s.variance_reduction = VarianceReduction::NONE;
            else
                throw std::runtime_error("sim_params: 无法识别的 variance_reduction = " + v);
        } else
            unknown.push_back(k);
    }
    warn_unknown_keys(path, unknown);

    // -------- 参数合法性校验 --------
    if (s.num_paths <= 0)
        throw std::runtime_error("sim_params 校验失败: num_paths 必须为正");
    if (s.num_steps <= 0)
        throw std::runtime_error("sim_params 校验失败: num_steps 必须为正");
    if (s.block_size <= 0)
        throw std::runtime_error("sim_params 校验失败: block_size 必须为正");
    return s;
}

// -------- CSV 输出 --------
void write_result_csv(const std::string& path, const OptionParams& o, const SimParams& s,
                      const PricingResult& r, double occupancy_pct, const GreeksBS* g) {
    std::ofstream of(path);
    of << std::fixed << std::setprecision(8);
    of << "option_type,price,ref_price,abs_error,std_error,ci_half,gpu_time_ms,cpu_time_ms,paths_"
          "per_sec,speedup,occupancy_pct,num_paths,num_steps,seed,variance_reduction,delta,gamma,"
          "vega,theta,rho\n";
    of << r.option_type_str << "," << r.price << "," << r.ref_price << "," << r.abs_error << ","
       << r.std_error << "," << r.ci_half << "," << r.gpu_time_ms << "," << r.cpu_time_ms << ","
       << r.paths_per_sec << "," << r.speedup << "," << occupancy_pct << "," << s.num_paths << ","
       << s.num_steps << "," << s.seed << "," << static_cast<int>(s.variance_reduction) << ",";
    if (g)
        of << g->Delta << "," << g->Gamma << "," << g->Vega << "," << g->Theta << "," << g->Rho;
    else
        of << ",,,,";
    of << "\n";
}

// -------- JSON 输出 --------
void write_result_json(const std::string& path, const OptionParams& o, const SimParams& s,
                       const PricingResult& r, double occupancy_pct, const GreeksBS* g) {
    std::ofstream of(path);
    of << std::fixed << std::setprecision(8);
    of << "{\n";
    // ---- option ----
    of << "  \"option\": {\n";
    of << "    \"type\": \"" << r.option_type_str << "\",\n";
    of << "    \"spot\": " << o.spot << ",\n";
    of << "    \"strike\": " << o.strike << ",\n";
    of << "    \"risk_free\": " << o.risk_free << ",\n";
    of << "    \"volatility\": " << o.volatility << ",\n";
    of << "    \"maturity\": " << o.maturity;
    if (o.type == OptionType::BARRIER_CALL || o.type == OptionType::BARRIER_PUT) {
        of << ",\n    \"barrier\": " << o.barrier << ",";
        of << "\n    \"barrier_dir\": \"" << (o.barrier_dir == BarrierDir::UP ? "up" : "down")
           << "\",";
        of << "\n    \"barrier_type\": \""
           << (o.barrier_t == BarrierType::KNOCK_IN ? "knock_in" : "knock_out") << "\"";
    }
    of << "\n  },\n";
    // ---- simulation ----
    of << "  \"simulation\": {\n";
    of << "    \"num_paths\": " << s.num_paths << ",\n";
    of << "    \"num_steps\": " << s.num_steps << ",\n";
    of << "    \"seed\": " << s.seed << ",\n";
    of << "    \"block_size\": " << s.block_size << ",\n";
    of << "    \"variance_reduction\": \"";
    switch (s.variance_reduction) {
    case VarianceReduction::NONE:
        of << "none";
        break;
    case VarianceReduction::ANTITHETIC:
        of << "antithetic";
        break;
    case VarianceReduction::CONTROL_VARIATE:
        of << "control_variate";
        break;
    }
    of << "\",\n    \"rng\": \"";
    of << "curand_philox";
    of << "\"\n  },\n";
    // ---- result ----
    of << "  \"result\": {\n";
    of << "    \"price\": " << r.price << ",\n";
    of << "    \"ref_price\": " << r.ref_price << ",\n";
    of << "    \"abs_error\": " << r.abs_error << ",\n";
    of << "    \"std_error\": " << r.std_error << ",\n";
    of << "    \"ci_half_95\": " << r.ci_half << ",\n";
    of << "    \"method\": \"" << r.method_str << "\"\n";
    of << "  }";
    // ---- greeks (可选) ----
    if (g) {
        of << ",\n  \"greeks\": {\n";
        of << "    \"delta\": " << g->Delta << ",\n";
        of << "    \"gamma\": " << g->Gamma << ",\n";
        of << "    \"vega\": " << g->Vega << ",\n";
        of << "    \"theta\": " << g->Theta << ",\n";
        of << "    \"rho\": " << g->Rho << "\n";
        of << "  }";
    }
    // ---- performance ----
    of << ",\n  \"performance\": {\n";
    of << "    \"gpu_time_ms\": " << r.gpu_time_ms << ",\n";
    of << "    \"cpu_time_ms\": " << r.cpu_time_ms << ",\n";
    of << "    \"paths_per_sec\": " << r.paths_per_sec << ",\n";
    of << "    \"speedup\": ";
    if (r.speedup >= 0)
        of << r.speedup;
    else
        of << "\"N/A\"";
    of << ",\n";
    of << "    \"occupancy_pct\": " << occupancy_pct << "\n";
    of << "  }\n";
    of << "}\n";
}

// -------- 性能日志 --------
void write_perf_log(const std::string& path, const OptionParams& o, const SimParams& s,
                    const PricingResult& r, double occupancy_pct, const GreeksBS* g /*=nullptr*/) {
    std::ofstream of(path, std::ios::app);
    of << std::fixed << std::setprecision(4);
    of << "[" << r.option_type_str << "]"
       << " paths=" << s.num_paths << " steps=" << s.num_steps << " seed=" << s.seed
       << " vr=" << static_cast<int>(s.variance_reduction) << " | price=" << r.price
       << " ref=" << r.ref_price << " SE=" << r.std_error << " | gpu=" << r.gpu_time_ms << "ms"
       << " cpu=" << r.cpu_time_ms << "ms"
       << " pps=" << r.paths_per_sec << " speedup=" << r.speedup << " occupancy=" << occupancy_pct
       << "%";
    if (g)
        of << " Delta=" << g->Delta << " Gamma=" << g->Gamma << " Vega=" << g->Vega
           << " Theta=" << g->Theta << " Rho=" << g->Rho;
    of << "\n";
}
