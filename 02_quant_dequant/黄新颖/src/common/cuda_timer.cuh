#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace quant_dequant::common {

/** @brief 表示 CUDA Event 创建、记录或同步失败时的错误。 */
class CudaTimerError final : public std::runtime_error {
public:
    /**
     * @brief 构造带有 CUDA runtime 诊断信息的计时器错误。
     *
     * @param detail 失败操作及 CUDA 错误文本。
     */
    explicit CudaTimerError(std::string detail);
};

/**
 * @brief 只测量指定 CUDA stream 中 kernel 区间的 move-only Event 计时器。
 *
 * 调用顺序必须是 `start()`、发射一个或多个 kernel、`stop()`、
 * `elapsedMilliseconds()`。调用方应把 H2D、D2H、Thrust 分配和文件 I/O 放在
 * start/stop 区间之外；stop event 会在同一 stream 中等待此前 kernel 完成，因此
 * elapsed 只包含两个 event 之间的 device 工作时间。
 */
class CudaEventTimer final {
public:
    /**
     * @brief 为指定 stream 创建一对 CUDA event。
     *
     * @param stream 要测量的非拥有 CUDA stream；默认 stream 传 nullptr。
     * @throws CudaTimerError CUDA event 创建失败时抛出。
     */
    explicit CudaEventTimer(cudaStream_t stream = nullptr);

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&&) = delete;
    CudaEventTimer& operator=(CudaEventTimer&&) = delete;

    /** @brief 销毁内部 CUDA event；析构函数不抛出异常。 */
    ~CudaEventTimer();

    /**
     * @brief 在关联 stream 中记录开始 event。
     *
     * 已完成一次 stop/elapsed 后可重新 start，计时器会覆盖旧区间。
     *
     * @throws CudaTimerError event 记录失败或上一个区间尚未 stop 时抛出。
     */
    void start();

    /**
     * @brief 在关联 stream 中记录结束 event。
     *
     * @throws CudaTimerError 尚未 start 或 event 记录失败时抛出。
     */
    void stop();

    /**
     * @brief 同步结束 event 并返回最近区间的毫秒数。
     *
     * @return 有限且非负的 CUDA Event elapsed time，单位毫秒。
     * @throws CudaTimerError 尚未 stop、同步或 elapsed 计算失败时抛出。
     */
    [[nodiscard]] float elapsedMilliseconds() const;

private:
    enum class State {
        /** 尚未记录开始 event，或已读取完前一次结果。 */
        kIdle,

        /** 已记录开始 event，尚未记录结束 event。 */
        kRunning,

        /** 已记录结束 event，可读取 elapsed time。 */
        kStopped,
    };

    /** 由对象拥有的开始 event。 */
    cudaEvent_t mStartEvent{};

    /** 由对象拥有的结束 event。 */
    cudaEvent_t mStopEvent{};

    /** 不拥有的、由调用方管理生命周期的计时 stream。 */
    cudaStream_t mStream{};

    /** 用于验证 API 调用顺序的当前状态。 */
    State mState{State::kIdle};
};

}  // namespace quant_dequant::common
