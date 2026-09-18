#include "common/cuda_timer.cuh"

#include <string>
#include <utility>

namespace quant_dequant::common {
namespace {

/**
 * @brief 将 CUDA runtime 状态转化为计时器异常。
 *
 * @param status CUDA runtime API 的返回值。
 * @param operation 失败的 CUDA event 操作名称。
 * @throws CudaTimerError status 不是 cudaSuccess 时抛出。
 */
void check_cuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        throw CudaTimerError{
            "CUDA Event 计时器 " + std::string{operation} + " 失败：" +
            cudaGetErrorString(status)};
    }
}

}  // namespace

CudaTimerError::CudaTimerError(std::string detail)
    : std::runtime_error(std::move(detail)) {}

CudaEventTimer::CudaEventTimer(const cudaStream_t stream)
    : mStream(stream) {
    check_cuda(cudaEventCreate(&mStartEvent), "创建 start event");
    try {
        check_cuda(cudaEventCreate(&mStopEvent), "创建 stop event");
    } catch (...) {
        // 构造函数第二个资源失败时主动回收第一个；析构函数尚不会运行。
        static_cast<void>(cudaEventDestroy(mStartEvent));
        mStartEvent = nullptr;
        throw;
    }
}

CudaEventTimer::~CudaEventTimer() {
    // 析构函数不能抛异常。此处只回收对象拥有的 event；若 CUDA context 已被外部
    // 销毁，runtime 可能报错，但没有可靠的析构期恢复动作。
    if (mStartEvent != nullptr) {
        static_cast<void>(cudaEventDestroy(mStartEvent));
    }
    if (mStopEvent != nullptr) {
        static_cast<void>(cudaEventDestroy(mStopEvent));
    }
}

void CudaEventTimer::start() {
    if (mState == State::kRunning) {
        throw CudaTimerError{"CUDA Event 计时器不能在未 stop 的区间内再次 start。"};
    }

    check_cuda(cudaEventRecord(mStartEvent, mStream), "记录 start event");
    mState = State::kRunning;
}

void CudaEventTimer::stop() {
    if (mState != State::kRunning) {
        throw CudaTimerError{"CUDA Event 计时器必须在 start 后才能 stop。"};
    }

    check_cuda(cudaEventRecord(mStopEvent, mStream), "记录 stop event");
    mState = State::kStopped;
}

float CudaEventTimer::elapsedMilliseconds() const {
    if (mState != State::kStopped) {
        throw CudaTimerError{
            "CUDA Event 计时器必须在 stop 后才能读取 elapsed time。"};
    }

    check_cuda(cudaEventSynchronize(mStopEvent), "同步 stop event");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, mStartEvent, mStopEvent),
               "计算 elapsed time");
    return elapsed_ms;
}

}  // namespace quant_dequant::common
