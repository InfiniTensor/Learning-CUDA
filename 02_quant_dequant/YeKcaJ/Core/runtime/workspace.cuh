#pragma once

// 任务显存与上传/下载接口；启动实现见 workspace.cu，不依赖文件读写或 CPU 校验。
#include "common/types.h"
#include "common/output_type.cuh"
#include "runtime/cuda_utils.cuh"

namespace pipeline {

// 显存及 event 与一次任务同寿命；resident benchmark 在重复间复用，不夹入传输。
struct Workspace {
  // n：元素数；group：每组元素数；groups：scale 个数；partials：全局归约块数。
  std::size_t n, group, groups;
  unsigned partials;
  Options options;
  // scratch 存局部最大值；state[0] 存全局最大值，state[1] 存 global_scale。
  DeviceBuffer<float> input, scratch, state;
  DeviceBuffer<std::uint8_t> data, scales;
  // 按 FP32 预留容量，FP16/BF16 输出也复用这块显存。
  DeviceBuffer<float> output;
  Events events;

  Workspace(std::size_t count, Options o)
      : n(count),
        group(o.tensor ? std::max<std::size_t>(1, n) : o.block),
        groups(ceil_div(n, group)),
        // 归约块上限从 4096 降到 1024；每个线程多做几次 grid-stride 扫描，
        // 保持 block 内分工和最大值结果不变，同时减少 partial/finalize 的调度负担。
        partials(static_cast<unsigned>(std::min<std::size_t>(1024, ceil_div(n, 256)))),
        options(o),
        input(n),
        scratch(partials),
        state(2),
        data(data_size(n)),
        scales(groups),
        output(n) {
    validate(o);
  }

  void upload(const std::vector<float>& x) {
    if (x.size() != n) throw std::runtime_error("workspace input length mismatch");
    if (n) CUDA_CHECK(cudaMemcpy(input.ptr, x.data(), n * 4, cudaMemcpyHostToDevice));
  }

  // 量化入口：全局最大值（按需）-> scale -> 元素编码/打包。
  // 同一默认 stream 按顺序执行，阶段之间不需要把 scale 拷回 CPU。
  void launch_quant(bool baseline = false);

  template <typename T>
  void launch_dequant();

  // 下载量化结果；文件写入由 app/run.cu 和 io/tensor_io.h 负责。
  Packed download(const Tensor& t) {
    Packed q{t.rows, t.cols, group, options, 1,
             std::vector<std::uint8_t>(data_size(n)),
             std::vector<std::uint8_t>(groups)};
    if (n) {
      CUDA_CHECK(cudaMemcpy(q.data.data(), data.ptr, q.data.size(), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(q.scales.data(), scales.ptr, groups, cudaMemcpyDeviceToHost));
      if (nvfp4) CUDA_CHECK(cudaMemcpy(&q.global, state.ptr + 1, 4, cudaMemcpyDeviceToHost));
    }
    return q;
  }

  // 上传已有低精度数据，用于不经过量化、直接从文件反量化的流程。
  void upload(const Packed& q) {
    if (checked_count(q.rows, q.cols) != n || q.group != group)
      throw std::runtime_error("workspace packed mismatch");

    const float s[2] = {0, q.global};
    CUDA_CHECK(cudaMemcpy(state.ptr, s, 8, cudaMemcpyHostToDevice));
    if (n) {
      CUDA_CHECK(cudaMemcpy(data.ptr, q.data.data(), q.data.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(scales.ptr, q.scales.data(), groups, cudaMemcpyHostToDevice));
    }
  }

  template <typename T>
  std::vector<T> download_output() {
    std::vector<T> y(n);
    if (n) CUDA_CHECK(cudaMemcpy(y.data(), output.ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return y;
  }
};

}  // namespace pipeline
