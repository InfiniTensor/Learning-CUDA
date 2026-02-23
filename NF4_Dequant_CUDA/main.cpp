#include "nf4_dequant.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

namespace {

struct RunParams {
    bool has_blocksize = false;
    int blocksize = 0;
    bool use_bf16 = true;
    std::string target_gpu = "T4";
    int blocks_per_group = 256;
    int block_dim = 256;
    bool autotune_block_dim = false;
    int autotune_repeats = 5;
    int kernel_warmup_iters = 0;
    int profile_loop_iters = 1;
    bool reuse_device_buffers = true;
    bool use_pinned_host_output = true;
    bool pinned_host_output_active = false;
    float autotune_ms_256 = -1.0f;
    float autotune_ms_512 = -1.0f;
    float bnb_time_ms = -1.0f;
    std::string perf_log_path;
};

std::string trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }
    return s.substr(begin, end - begin);
}

std::string unquote(std::string value) {
    value = trim(value);
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

bool parse_int(const std::string& s, int* out) {
    if (out == nullptr) {
        return false;
    }
    try {
        size_t pos = 0;
        const int v = std::stoi(s, &pos, 10);
        if (pos != s.size()) {
            return false;
        }
        *out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_float(const std::string& s, float* out) {
    if (out == nullptr) {
        return false;
    }
    try {
        size_t pos = 0;
        const float v = std::stof(s, &pos);
        if (pos != s.size()) {
            return false;
        }
        *out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_bool(const std::string& s, bool* out) {
    if (out == nullptr) {
        return false;
    }
    std::string t = trim(s);
    for (char& c : t) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (t == "1" || t == "true" || t == "yes" || t == "on") {
        *out = true;
        return true;
    }
    if (t == "0" || t == "false" || t == "no" || t == "off") {
        *out = false;
        return true;
    }
    return false;
}

bool load_params_file(const char* params_path, RunParams* params) {
    if (params_path == nullptr || params == nullptr) {
        return false;
    }

    std::ifstream fin(params_path);
    if (!fin.is_open()) {
        std::fprintf(stderr, "Failed to open params file: %s\n", params_path);
        return false;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(fin, line)) {
        ++line_no;
        const size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            std::fprintf(stderr, "params parse error at line %d: missing '='\n", line_no);
            return false;
        }

        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        value = unquote(value);

        if (key == "blocksize") {
            int v = 0;
            if (!parse_int(value, &v) || v <= 0) {
                std::fprintf(stderr, "params parse error at line %d: invalid blocksize\n", line_no);
                return false;
            }
            params->blocksize = v;
            params->has_blocksize = true;
        } else if (key == "compute_type") {
            if (value == "bf16") {
                params->use_bf16 = true;
            } else if (value == "fp16") {
                params->use_bf16 = false;
            } else {
                std::fprintf(stderr, "params parse error at line %d: compute_type must be bf16/fp16\n", line_no);
                return false;
            }
        } else if (key == "target_gpu") {
            params->target_gpu = value;
        } else if (key == "blocks_per_group") {
            int v = 0;
            if (!parse_int(value, &v) || v <= 0) {
                std::fprintf(stderr, "params parse error at line %d: invalid blocks_per_group\n", line_no);
                return false;
            }
            params->blocks_per_group = v;
        } else if (key == "block_dim") {
            int v = 0;
            if (!parse_int(value, &v) || v <= 0 || v > 1024) {
                std::fprintf(stderr, "params parse error at line %d: invalid block_dim\n", line_no);
                return false;
            }
            params->block_dim = v;
        } else if (key == "autotune_block_dim") {
            bool v = false;
            if (!parse_bool(value, &v)) {
                std::fprintf(stderr, "params parse error at line %d: invalid autotune_block_dim\n", line_no);
                return false;
            }
            params->autotune_block_dim = v;
        } else if (key == "autotune_repeats") {
            int v = 0;
            if (!parse_int(value, &v) || v <= 0 || v > 1000) {
                std::fprintf(stderr, "params parse error at line %d: invalid autotune_repeats\n", line_no);
                return false;
            }
            params->autotune_repeats = v;
        } else if (key == "kernel_warmup_iters") {
            int v = 0;
            if (!parse_int(value, &v) || v < 0 || v > 10000) {
                std::fprintf(stderr, "params parse error at line %d: invalid kernel_warmup_iters\n", line_no);
                return false;
            }
            params->kernel_warmup_iters = v;
        } else if (key == "profile_loop_iters") {
            int v = 0;
            if (!parse_int(value, &v) || v <= 0 || v > 10000) {
                std::fprintf(stderr, "params parse error at line %d: invalid profile_loop_iters\n", line_no);
                return false;
            }
            params->profile_loop_iters = v;
        } else if (key == "reuse_device_buffers") {
            bool v = false;
            if (!parse_bool(value, &v)) {
                std::fprintf(stderr, "params parse error at line %d: invalid reuse_device_buffers\n", line_no);
                return false;
            }
            params->reuse_device_buffers = v;
        } else if (key == "use_pinned_host_output") {
            bool v = false;
            if (!parse_bool(value, &v)) {
                std::fprintf(stderr, "params parse error at line %d: invalid use_pinned_host_output\n", line_no);
                return false;
            }
            params->use_pinned_host_output = v;
        } else if (key == "bnb_time_ms") {
            float v = -1.0f;
            if (!parse_float(value, &v) || v <= 0.0f) {
                std::fprintf(stderr, "params parse error at line %d: invalid bnb_time_ms\n", line_no);
                return false;
            }
            params->bnb_time_ms = v;
        } else if (key == "perf_log_path") {
            params->perf_log_path = value;
        }
    }

    return true;
}

double compute_effective_bandwidth_gbps(const NF4QuantState& state, float kernel_ms) {
    if (kernel_ms <= 0.0f) {
        return 0.0;
    }
    const double bytes_read =
        static_cast<double>(state.num_packed_bytes) +
        static_cast<double>(state.num_blocks) * sizeof(uint8_t) +
        static_cast<double>(state.num_groups) * sizeof(__half) +
        256.0 * sizeof(__half);
    const double bytes_written = static_cast<double>(state.num_elements) * sizeof(uint16_t);
    const double total_bytes = bytes_read + bytes_written;
    const double seconds = static_cast<double>(kernel_ms) * 1e-3;
    return total_bytes / seconds / 1e9;
}

std::string default_perf_log_path(const char* output_path) {
    return std::string(output_path) + ".perf.log";
}

bool write_perf_log(
    const std::string& log_path,
    const NF4QuantState& state,
    const RunParams& params,
    float kernel_ms,
    double effective_bw_gbps,
    double speedup_vs_bnb) {
    std::ofstream fout(log_path, std::ios::out | std::ios::trunc);
    if (!fout.is_open()) {
        std::fprintf(stderr, "Warning: failed to write perf log: %s\n", log_path.c_str());
        return false;
    }

    fout.setf(std::ios::fixed);
    fout.precision(6);

    fout << "kernel_time_ms=" << kernel_ms << "\n";
    fout << "effective_bandwidth_gbps=" << effective_bw_gbps << "\n";
    if (std::isfinite(speedup_vs_bnb)) {
        fout << "speedup_vs_bnb=" << speedup_vs_bnb << "\n";
    } else {
        fout << "speedup_vs_bnb=N/A\n";
    }
    fout << "target_gpu=" << params.target_gpu << "\n";
    fout << "compute_type=" << (params.use_bf16 ? "bf16" : "fp16") << "\n";
    fout << "blocksize=" << state.blocksize << "\n";
    fout << "blocks_per_group=" << state.blocks_per_group << "\n";
    fout << "block_dim=" << params.block_dim << "\n";
    fout << "autotune_block_dim=" << (params.autotune_block_dim ? "true" : "false") << "\n";
    fout << "autotune_repeats=" << params.autotune_repeats << "\n";
    fout << "kernel_warmup_iters=" << params.kernel_warmup_iters << "\n";
    fout << "profile_loop_iters=" << params.profile_loop_iters << "\n";
    fout << "reuse_device_buffers=" << (params.reuse_device_buffers ? "true" : "false") << "\n";
    fout << "use_pinned_host_output=" << (params.use_pinned_host_output ? "true" : "false") << "\n";
    fout << "pinned_host_output_active=" << (params.pinned_host_output_active ? "true" : "false") << "\n";
    if (params.autotune_ms_256 > 0.0f) {
        fout << "autotune_ms_256=" << params.autotune_ms_256 << "\n";
    }
    if (params.autotune_ms_512 > 0.0f) {
        fout << "autotune_ms_512=" << params.autotune_ms_512 << "\n";
    }
    fout << "num_rows=" << state.num_rows << "\n";
    fout << "num_cols=" << state.num_cols << "\n";
    fout << "num_elements=" << state.num_elements << "\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <weights.bin> <params.txt> <output.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* weights_path = argv[1];
    const char* params_path = argv[2];
    const char* output_path = argv[3];

    RunParams params{};
    if (!load_params_file(params_path, &params)) {
        return EXIT_FAILURE;
    }

    NF4QuantState state{};
    if (!load_nf4_file(weights_path, &state)) {
        std::fprintf(stderr, "Failed to load NF4 file: %s\n", weights_path);
        return EXIT_FAILURE;
    }

    if (params.has_blocksize && params.blocksize != state.blocksize) {
        std::fprintf(stderr,
            "Warning: blocksize mismatch (file=%d, params=%d). Using file blocksize=%d.\n",
            state.blocksize,
            params.blocksize,
            state.blocksize);
    }
    state.blocks_per_group = params.blocks_per_group;

    const size_t output_bytes = state.num_elements * sizeof(uint16_t);
    std::vector<uint16_t> output_pageable;
    uint16_t* output_ptr = nullptr;
    if (params.use_pinned_host_output) {
        cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&output_ptr), output_bytes);
        if (err == cudaSuccess) {
            params.pinned_host_output_active = true;
        } else {
            std::fprintf(
                stderr,
                "Warning: cudaMallocHost failed for output buffer (%s). Falling back to pageable host memory.\n",
                cudaGetErrorString(err));
        }
    }
    if (output_ptr == nullptr) {
        output_pageable.assign(state.num_elements, 0U);
        output_ptr = output_pageable.data();
        params.pinned_host_output_active = false;
    }

    auto free_output_buffer = [&]() {
        if (params.pinned_host_output_active && output_ptr != nullptr) {
            cudaFreeHost(output_ptr);
            output_ptr = nullptr;
            params.pinned_host_output_active = false;
        }
    };

    auto run_once = [&](void* out_ptr, int block_dim, float* out_ms, bool copy_output_to_host) -> bool {
        return cuda_dequant_nf4(
            state,
            out_ptr,
            params.use_bf16,
            out_ms,
            block_dim,
            copy_output_to_host,
            params.reuse_device_buffers);
    };

    int selected_block_dim = params.block_dim;
    if (params.autotune_block_dim) {
        constexpr int candidates[2] = {256, 512};
        double best_ms = std::numeric_limits<double>::infinity();
        int best_dim = candidates[0];

        for (int cand : candidates) {
            float warmup_ms = 0.0f;
            if (!run_once(nullptr, cand, &warmup_ms, false)) {
                std::fprintf(stderr, "CUDA dequant warmup failed for block_dim=%d.\n", cand);
                free_output_buffer();
                cuda_release_nf4_device_cache();
                free_nf4_state(&state);
                return EXIT_FAILURE;
            }

            double total_ms = 0.0;
            for (int rep = 0; rep < params.autotune_repeats; ++rep) {
                float ms = 0.0f;
                if (!run_once(nullptr, cand, &ms, false)) {
                    std::fprintf(stderr, "CUDA dequant failed during autotune for block_dim=%d.\n", cand);
                    free_output_buffer();
                    cuda_release_nf4_device_cache();
                    free_nf4_state(&state);
                    return EXIT_FAILURE;
                }
                total_ms += static_cast<double>(ms);
            }

            const float avg_ms = static_cast<float>(total_ms / static_cast<double>(params.autotune_repeats));
            if (cand == 256) {
                params.autotune_ms_256 = avg_ms;
            } else if (cand == 512) {
                params.autotune_ms_512 = avg_ms;
            }
            std::printf(
                "AutoTune candidate blockDim=%d avg_kernel_ms=%.4f (%d reps)\n",
                cand,
                avg_ms,
                params.autotune_repeats);

            if (avg_ms < best_ms) {
                best_ms = avg_ms;
                best_dim = cand;
            }
        }

        selected_block_dim = best_dim;
        params.block_dim = best_dim;
        std::printf("AutoTune selected blockDim=%d\n", selected_block_dim);
    }

    for (int i = 0; i < params.kernel_warmup_iters; ++i) {
        float warmup_ms = 0.0f;
        if (!run_once(nullptr, selected_block_dim, &warmup_ms, false)) {
            std::fprintf(stderr, "CUDA dequant failed during kernel_warmup_iters at iter=%d.\n", i);
            free_output_buffer();
            cuda_release_nf4_device_cache();
            free_nf4_state(&state);
            return EXIT_FAILURE;
        }
    }

    float kernel_ms = 0.0f;
    bool profiler_started = false;
    const int profile_loop_iters = params.profile_loop_iters > 0 ? params.profile_loop_iters : 1;
    for (int iter = 0; iter < profile_loop_iters; ++iter) {
        if (iter == 1 && profile_loop_iters > 1) {
            const cudaError_t prof_err = cudaProfilerStart();
            if (prof_err != cudaSuccess) {
                std::fprintf(stderr, "Warning: cudaProfilerStart failed: %s\n", cudaGetErrorString(prof_err));
            } else {
                profiler_started = true;
            }
        }

        if (!run_once(output_ptr, selected_block_dim, &kernel_ms, true)) {
            std::fprintf(stderr, "CUDA dequant failed at profile loop iter=%d.\n", iter);
            if (profiler_started) {
                cudaProfilerStop();
            }
            free_output_buffer();
            cuda_release_nf4_device_cache();
            free_nf4_state(&state);
            return EXIT_FAILURE;
        }
    }
    if (profiler_started) {
        const cudaError_t prof_stop_err = cudaProfilerStop();
        if (prof_stop_err != cudaSuccess) {
            std::fprintf(stderr, "Warning: cudaProfilerStop failed: %s\n", cudaGetErrorString(prof_stop_err));
        }
    }

    save_dequant(output_ptr, state.num_rows, state.num_cols, output_path, params.use_bf16);

    const double bw_gbps = compute_effective_bandwidth_gbps(state, kernel_ms);
    std::printf("Host output buffer: %s\n", params.pinned_host_output_active ? "pinned" : "pageable");
    std::printf("Reuse device buffers: %s\n", params.reuse_device_buffers ? "true" : "false");
    if (profile_loop_iters > 1) {
        std::printf("Profile loop iters: %d (capture last %d)\n", profile_loop_iters, profile_loop_iters - 1);
    } else {
        std::printf("Profile loop iters: 1 (single run)\n");
    }
    std::printf("BlockDim: %d\n", selected_block_dim);
    std::printf("Kernel time: %.4f ms\n", kernel_ms);
    std::printf("Effective bandwidth: %.2f GB/s\n", bw_gbps);
    double speedup = std::numeric_limits<double>::quiet_NaN();
    if (params.bnb_time_ms > 0.0f) {
        speedup = static_cast<double>(params.bnb_time_ms) / static_cast<double>(kernel_ms);
        std::printf("Speedup vs bnb: %.2fx\n", speedup);
    } else {
        std::printf("Speedup vs bnb: N/A (set bnb_time_ms in params.txt)\n");
    }

    const std::string log_path =
        params.perf_log_path.empty() ? default_perf_log_path(output_path) : params.perf_log_path;
    if (write_perf_log(log_path, state, params, kernel_ms, bw_gbps, speedup)) {
        std::printf("Perf log: %s\n", log_path.c_str());
    }

    free_output_buffer();
    cuda_release_nf4_device_cache();
    free_nf4_state(&state);
    return EXIT_SUCCESS;
}
