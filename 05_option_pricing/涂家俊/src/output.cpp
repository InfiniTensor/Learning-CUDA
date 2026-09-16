// 将定价和性能结果安全地写为 JSON 与 CSV 文件。
#include "pricer/output.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace pricer {
namespace {

constexpr const char *k_csv_header =
    "schema_version,run_id,timestamp_utc,option_type,backend,precision,"
    "num_paths,num_steps,batch_size,seed,block_size,grid_size,"
    "total_runtime_ms,compute_runtime_ms,rng_ms,reduction_ms,"
    "paths_per_second,cpu_runtime_ms,speedup_vs_cpu,gpu_name,"
    "cuda_version,git_commit";

const char *option_type_name(OptionType type) {
    switch (type) {
    case OptionType::EuropeanCall:
        return "european_call";
    case OptionType::AsianArithmeticCall:
        return "asian_call";
    }
    throw OutputError("unknown option type");
}

const char *precision_name(Precision precision) {
    switch (precision) {
    case Precision::Fp32:
        return "fp32";
    case Precision::Fp64:
        return "fp64";
    }
    throw OutputError("unknown precision");
}

const char *rng_name(RngType rng) {
    switch (rng) {
    case RngType::CurandPhilox:
        return "curand_philox";
    case RngType::Mt19937_64:
        return "mt19937_64";
    }
    throw OutputError("unknown RNG");
}

const char *variance_reduction_name(VarianceReduction reduction) {
    if (reduction == VarianceReduction::None) {
        return "none";
    }
    throw OutputError("unknown variance reduction");
}

template <typename Value>
nlohmann::json optional_json(const std::optional<Value> &value) {
    return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

std::string numeric(double value) {
    if (!std::isfinite(value)) {
        return {};
    }
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

std::string optional_numeric(const std::optional<double> &value) {
    return value ? numeric(*value) : std::string{};
}

std::string csv_escape(const std::optional<std::string> &value) {
    if (!value) {
        return {};
    }
    if (value->find_first_of(",\"\r\n") == std::string::npos) {
        return *value;
    }
    std::string escaped = "\"";
    for (const char character : *value) {
        escaped += character;
        if (character == '"') {
            escaped += '"';
        }
    }
    escaped += '"';
    return escaped;
}

std::string csv_escape(const std::string &value) {
    return csv_escape(std::optional<std::string>(value));
}

void require_output_directory(const std::filesystem::path &directory) {
    std::error_code error;
    if (std::filesystem::create_directories(directory, error)) {
        return;
    }
    if (error) {
        throw OutputError("could not create output directory: " +
                          directory.string() + ": " + error.message());
    }
    const bool is_directory = std::filesystem::is_directory(directory, error);
    if (error) {
        throw OutputError("could not inspect output directory: " +
                          directory.string() + ": " + error.message());
    }
    if (!is_directory) {
        throw OutputError("output path is not a directory: " +
                          directory.string());
    }
}

class OutputLock {
  public:
    explicit OutputLock(const std::filesystem::path &directory) {
        require_output_directory(directory);
        path_ = directory / ".pricer-output.lock";
#ifdef _WIN32
        for (std::uint32_t attempt = 0U; attempt < 5000U; ++attempt) {
            handle_ = CreateFileW(path_.c_str(), GENERIC_READ | GENERIC_WRITE,
                                  0, nullptr, OPEN_ALWAYS,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
            if (handle_ != INVALID_HANDLE_VALUE) {
                return;
            }
            const DWORD error = GetLastError();
            if (error != ERROR_SHARING_VIOLATION &&
                error != ERROR_LOCK_VIOLATION) {
                throw OutputError(
                    "could not acquire output lock: Windows error " +
                    std::to_string(error));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw OutputError("timed out acquiring output lock: " + path_.string());
#else
        descriptor_ = open(path_.c_str(), O_CREAT | O_RDWR, 0666);
        if (descriptor_ < 0) {
            throw OutputError("could not open output lock: " +
                              std::string(std::strerror(errno)));
        }
        if (flock(descriptor_, LOCK_EX) != 0) {
            const std::string reason = std::strerror(errno);
            close(descriptor_);
            descriptor_ = -1;
            throw OutputError("could not acquire output lock: " + reason);
        }
#endif
    }

    OutputLock(const OutputLock &) = delete;
    OutputLock &operator=(const OutputLock &) = delete;

    ~OutputLock() noexcept {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
#else
        if (descriptor_ >= 0) {
            flock(descriptor_, LOCK_UN);
            close(descriptor_);
        }
#endif
    }

  private:
    std::filesystem::path path_;
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int descriptor_ = -1;
#endif
};

std::uint64_t process_id() {
#ifdef _WIN32
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

bool path_exists(const std::filesystem::path &path, const char *context) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        throw OutputError(std::string(context) + ": " + error.message());
    }
    return exists;
}

struct JsonWrite {
    std::filesystem::path path;
    std::string run_id;
};

struct CsvState {
    bool existed;
    std::uintmax_t size;
    bool needs_newline;
};

std::string csv_row(const OutputRecord &record) {
    std::ostringstream output;
    output << "1.0," << csv_escape(record.run_id) << ','
           << csv_escape(record.timestamp_utc) << ','
           << csv_escape(std::string(option_type_name(record.option.type)))
           << ',' << csv_escape(record.backend) << ','
           << csv_escape(
                  std::string(precision_name(record.simulation.precision)))
           << ',' << record.simulation.num_paths << ','
           << record.simulation.num_steps << ',' << record.simulation.batch_size
           << ',' << record.simulation.seed << ',';
    if (record.backend == "gpu") {
        output << record.performance.block_size;
    }
    output << ',';
    if (record.backend == "gpu") {
        output << record.performance.grid_size;
    }
    output << ',' << numeric(record.performance.total_runtime_ms) << ','
           << numeric(record.performance.compute_runtime_ms) << ','
           << optional_numeric(record.performance.rng_ms) << ','
           << optional_numeric(record.performance.reduction_ms) << ','
           << numeric(record.performance.paths_per_second) << ','
           << optional_numeric(record.performance.cpu_runtime_ms) << ','
           << optional_numeric(record.performance.speedup_vs_cpu) << ','
           << csv_escape(record.environment.gpu_name) << ','
           << csv_escape(record.environment.cuda_runtime_version) << ','
           << csv_escape(record.environment.git_commit);
    return output.str();
}

CsvState inspect_csv_unlocked(const std::filesystem::path &path) {
    std::error_code error;
    const bool existed = std::filesystem::exists(path, error);
    if (error) {
        throw OutputError("could not inspect performance CSV: " +
                          error.message());
    }
    if (!existed) {
        return {false, 0U, false};
    }
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        throw OutputError("could not inspect performance CSV size: " +
                          error.message());
    }
    if (size == 0U) {
        return {true, 0U, false};
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw OutputError("could not open performance CSV for validation: " +
                          path.string());
    }
    std::string header;
    if (!std::getline(input, header)) {
        throw OutputError("could not read performance CSV header");
    }
    if (!header.empty() && header.back() == '\r') {
        header.pop_back();
    }
    if (header != k_csv_header) {
        throw OutputError("performance CSV header does not match schema 1.0");
    }
    input.clear();
    input.seekg(-1, std::ios::end);
    char last = '\0';
    input.get(last);
    if (!input) {
        throw OutputError("could not inspect performance CSV tail");
    }
    return {true, size, last != '\n'};
}

void append_csv_unlocked(const std::filesystem::path &path,
                         const OutputRecord &record, const CsvState &state) {
    std::ofstream output(path, std::ios::binary | std::ios::app);
    if (!output) {
        throw OutputError("could not open performance CSV for append: " +
                          path.string());
    }
    if (state.size == 0U) {
        output << k_csv_header << '\n';
    } else if (state.needs_newline) {
        output << '\n';
    }
    output << csv_row(record) << '\n';
    output.flush();
    output.close();
    if (!output) {
        throw OutputError("could not append performance CSV: " + path.string());
    }
}

std::filesystem::path
unique_temporary_path(const std::filesystem::path &final_path) {
    static std::atomic<std::uint64_t> next_id{0U};
    for (std::uint32_t attempt = 0U; attempt < 100U; ++attempt) {
        const auto candidate = std::filesystem::path(
            final_path.string() + "." + std::to_string(process_id()) + "." +
            std::to_string(next_id.fetch_add(1U)) + ".tmp");
        if (!path_exists(candidate, "could not inspect temporary JSON path")) {
            return candidate;
        }
    }
    throw OutputError("could not reserve a unique temporary JSON path");
}

JsonWrite write_json_unlocked(const std::filesystem::path &output_dir,
                              OutputRecord record) {
    const std::string base_run_id =
        record.run_id.empty()
            ? record.timestamp_utc + "-" +
                  option_type_name(record.option.type) + "-" + record.backend
            : record.run_id;
    std::filesystem::path final_path =
        output_dir / (base_run_id + "-result.json");
    for (std::uint64_t suffix = 1U;
         path_exists(final_path, "could not inspect result JSON path");
         ++suffix) {
        final_path = output_dir / (base_run_id + "-" + std::to_string(suffix) +
                                   "-result.json");
    }
    std::string final_run_id = final_path.stem().string();
    final_run_id.resize(final_run_id.size() - std::string("-result").size());
    record.run_id = final_run_id;

    const auto temporary_path = unique_temporary_path(final_path);
    try {
        std::ofstream stream(temporary_path,
                             std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw OutputError("could not open temporary JSON: " +
                              temporary_path.string());
        }
        stream << make_result_json(record).dump(2) << '\n';
        stream.flush();
        if (!stream) {
            throw OutputError("could not write temporary JSON: " +
                              temporary_path.string());
        }
        stream.close();
        if (!stream) {
            throw OutputError("could not close temporary JSON: " +
                              temporary_path.string());
        }
        std::error_code error;
        std::filesystem::rename(temporary_path, final_path, error);
        if (error) {
            throw OutputError("could not publish result JSON: " +
                              error.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        throw;
    }
    return {final_path, final_run_id};
}

} // namespace

nlohmann::json make_result_json(const OutputRecord &record) {
    nlohmann::json json;
    const nlohmann::json confidence_interval =
        record.result.ci_lower && record.result.ci_upper
            ? nlohmann::json::array(
                  {*record.result.ci_lower, *record.result.ci_upper})
            : nlohmann::json(nullptr);
    json["schema_version"] = "1.0";
    json["run_id"] = record.run_id;
    json["status"] = "success";
    json["timestamp_utc"] = record.timestamp_utc;
    json["backend"] = record.backend;
    json["option"] = {{"type", option_type_name(record.option.type)},
                      {"spot", record.option.spot},
                      {"strike", record.option.strike},
                      {"risk_free_rate", record.option.risk_free_rate},
                      {"volatility", record.option.volatility},
                      {"maturity", record.option.maturity},
                      {"barrier", optional_json(record.option.barrier)}};
    json["simulation"] = {
        {"num_paths", record.simulation.num_paths},
        {"num_steps", record.simulation.num_steps},
        {"seed", record.simulation.seed},
        {"rng", rng_name(record.simulation.rng)},
        {"variance_reduction",
         variance_reduction_name(record.simulation.variance_reduction)},
        {"precision", precision_name(record.simulation.precision)},
        {"block_size", record.simulation.block_size},
        {"batch_size", record.simulation.batch_size}};
    json["result"] = {
        {"price_estimate", record.result.price},
        {"sample_stddev", optional_json(record.result.sample_stddev)},
        {"standard_error", optional_json(record.result.standard_error)},
        {"confidence_level", 0.95},
        {"confidence_interval", confidence_interval},
        {"reference_price", optional_json(record.result.reference_price)},
        {"absolute_error", optional_json(record.result.absolute_error)},
        {"relative_error", optional_json(record.result.relative_error)}};
    json["performance"] = {
        {"total_runtime_ms", record.performance.total_runtime_ms},
        {"compute_runtime_ms", record.performance.compute_runtime_ms},
        {"gpu_compute_ms",
         record.backend == "gpu"
             ? nlohmann::json(record.performance.compute_runtime_ms)
             : nlohmann::json(nullptr)},
        {"rng_ms", optional_json(record.performance.rng_ms)},
        {"reduction_ms", optional_json(record.performance.reduction_ms)},
        {"paths_per_second", record.performance.paths_per_second},
        {"block_size", record.backend == "gpu"
                           ? nlohmann::json(record.performance.block_size)
                           : nlohmann::json(nullptr)},
        {"grid_size", record.backend == "gpu"
                          ? nlohmann::json(record.performance.grid_size)
                          : nlohmann::json(nullptr)},
        {"repetitions", record.performance.repetitions},
        {"aggregation", record.performance.aggregation},
        {"cpu_runtime_ms", optional_json(record.performance.cpu_runtime_ms)},
        {"speedup_vs_cpu", optional_json(record.performance.speedup_vs_cpu)}};
    json["environment"] = {
        {"gpu_name", optional_json(record.environment.gpu_name)},
        {"compute_capability",
         optional_json(record.environment.compute_capability)},
        {"cuda_runtime_version",
         optional_json(record.environment.cuda_runtime_version)},
        {"build_type", optional_json(record.environment.build_type)},
        {"git_commit", optional_json(record.environment.git_commit)}};
    return json;
}

std::filesystem::path
write_result_json_atomic(const std::filesystem::path &output_dir,
                         OutputRecord record) {
    OutputLock lock(output_dir);
    return write_json_unlocked(output_dir, std::move(record)).path;
}

void append_performance_csv(const std::filesystem::path &path,
                            const OutputRecord &record) {
    const auto directory = path.parent_path().empty()
                               ? std::filesystem::path(".")
                               : path.parent_path();
    OutputLock lock(directory);
    append_csv_unlocked(path, record, inspect_csv_unlocked(path));
}

WrittenOutput write_output_transaction(const std::filesystem::path &output_dir,
                                       OutputRecord record) {
    OutputLock lock(output_dir);
    const auto csv_path = output_dir / "performance.csv";
    const CsvState csv_state = inspect_csv_unlocked(csv_path);
    const JsonWrite json = write_json_unlocked(output_dir, record);
    record.run_id = json.run_id;
    try {
        append_csv_unlocked(csv_path, record, csv_state);
    } catch (const std::exception &error) {
        std::error_code json_error;
        std::filesystem::remove(json.path, json_error);
        std::error_code csv_error;
        if (csv_state.existed) {
            std::filesystem::resize_file(csv_path, csv_state.size, csv_error);
        } else {
            std::filesystem::remove(csv_path, csv_error);
        }
        if (json_error || csv_error) {
            throw OutputError(std::string(error.what()) +
                              "; output rollback failed");
        }
        throw;
    }
    return {json.path, csv_path, json.run_id};
}

double median(std::vector<double> measurements) {
    if (measurements.empty()) {
        throw std::invalid_argument("median requires at least one measurement");
    }
    std::sort(measurements.begin(), measurements.end());
    const std::size_t middle = measurements.size() / 2U;
    if (measurements.size() % 2U != 0U) {
        return measurements[middle];
    }
    return (measurements[middle - 1U] + measurements[middle]) / 2.0;
}

std::string utc_timestamp() {
    const std::time_t now =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y%m%dT%H%M%SZ");
    return stream.str();
}

} // namespace pricer
