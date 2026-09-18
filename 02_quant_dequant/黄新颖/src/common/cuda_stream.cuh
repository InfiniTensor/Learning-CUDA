#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace quant_dequant::common {

/** @brief 表示 CUDA stream 创建或同步失败时的错误。 */
class CudaStreamError final : public std::runtime_error {
public:
    /**
     * @brief 构造带有 CUDA runtime 诊断的 stream 错误。
     *
     * @param detail 失败操作及 CUDA 错误文本。
     */
    explicit CudaStreamError(std::string detail);
};

/**
 * @brief 拥有一条 non-blocking CUDA stream 的 RAII 对象。
 *
 * stream 用于串联一次 pipeline 调用的 H2D、CUDA Event、全部 kernel 和 D2H。
 * 它使用 `cudaStreamNonBlocking` 创建，因此不因 legacy default stream 与其他
 * 独立 stream 发生隐式双向等待；本对象不跨线程共享，也不暴露所有权转移。
 */
class CudaStream final {
public:
    /**
     * @brief 创建一条 non-blocking CUDA stream。
     *
     * @throws CudaStreamError 当前 device 无法创建 stream 时抛出。
     */
    CudaStream();

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&&) = delete;
    CudaStream& operator=(CudaStream&&) = delete;

    /** @brief 销毁拥有的 CUDA stream；析构函数不抛出异常。 */
    ~CudaStream();

    /**
     * @brief 借出给 Thrust execution policy、event 和 kernel launch 使用的 stream。
     *
     * @return 不拥有的有效 cudaStream_t；其生命周期由本对象保证。
     */
    [[nodiscard]] cudaStream_t get() const noexcept;

    /**
     * @brief 等待当前 stream 中此前提交的全部工作完成。
     *
     * @throws CudaStreamError CUDA runtime 同步失败时抛出。
     */
    void synchronize() const;

private:
    /** 由本对象独占拥有的 CUDA stream。 */
    cudaStream_t mStream{};
};

}  // namespace quant_dequant::common
