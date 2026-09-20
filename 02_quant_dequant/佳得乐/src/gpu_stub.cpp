#include "project.hpp"
namespace lp {
Packed quant_gpu(const Tensor &, const Config &, const std::string &, Timing *, Tensor *) {
    throw std::runtime_error("CUDA disabled in this build");
}
Tensor dequant_gpu(const Packed &, Timing *) {
    throw std::runtime_error("CUDA disabled in this build");
}
} // namespace lp
