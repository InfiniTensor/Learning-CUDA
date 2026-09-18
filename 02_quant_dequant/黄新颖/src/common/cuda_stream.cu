#include "common/cuda_stream.cuh"

#include <string>
#include <utility>

namespace quant_dequant::common {
namespace {

/**
 * @brief 检查 CUDA runtime 返回值并转换为 CudaStreamError。
 *
 * @param status CUDA runtime API 状态。
 * @param operation 失败的 stream 操作名称。
 * @throws CudaStreamError status 不是 cudaSuccess 时抛出。
 */
void check_cuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        throw CudaStreamError{
            "CUDA stream " + std::string{operation} + " 失败：" +
            cudaGetErrorString(status)};
    }
}

}  // namespace

CudaStreamError::CudaStreamError(std::string detail)
    : std::runtime_error(std::move(detail)) {}

CudaStream::CudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking),
               "创建 non-blocking stream");
}

CudaStream::~CudaStream() {
    if (mStream != nullptr) {
        static_cast<void>(cudaStreamDestroy(mStream));
    }
}

cudaStream_t CudaStream::get() const noexcept {
    return mStream;
}

void CudaStream::synchronize() const {
    check_cuda(cudaStreamSynchronize(mStream), "同步 stream");
}

}  // namespace quant_dequant::common
