#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include "kernels.cuh"
#include "contiguous256.cuh"
#include "packed_rows.cuh"
#include "packed_pairs.cuh"
#include "row_policy.hpp"
#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

namespace {

void validate(const at::Tensor& input, double scale, int block_threads) {
    TORCH_CHECK(block_threads == 128 || block_threads == 256, "block_threads must be 128 or 256");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "input dtype must be float16 or bfloat16");
    TORCH_CHECK(input.dim() == 2 || input.dim() == 4, "input must have 2 or 4 dimensions");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(!input.is_neg(), "input must not have a lazy negative bit; call resolve_neg() first");
    TORCH_CHECK(input.numel() > 0, "input must be nonempty");
    const auto dim = input.size(-1);
    TORCH_CHECK(dim >= 1 && dim <= 256 && !(dim & (dim - 1)),
                "last dimension must be a power of two in [1,256]");
    TORCH_CHECK(!input.requires_grad(), "this forward-only extension does not support requires_grad=True");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 &&
                    std::isfinite(static_cast<float>(scale)) && static_cast<float>(scale) > 0,
                "scale must be finite, positive, and representable as float32");
    const auto rows = input.numel() / dim;
    TORCH_CHECK((rows - 1) / (block_threads / 32) + 1 <= std::numeric_limits<int>::max(),
                "input has too many rows for the CUDA launch grid");
}

template <class T, int N, bool Transform, bool Quantize>
void launch(const at::Tensor& input, at::Tensor& output, at::Tensor& packed,
            at::Tensor& scales, float scale, cudaStream_t stream, int block_threads,
            bool contiguous256_fused, bool packed_rows) {
    const auto rows = static_cast<std::size_t>(input.numel() / N);
    const auto blocks = static_cast<unsigned int>((rows - 1) / (block_threads / 32) + 1);
    const auto* source = reinterpret_cast<const T*>(input.data_ptr());
    auto* destination = output.defined() ? reinterpret_cast<T*>(output.data_ptr()) : nullptr;
    auto* bytes = packed.defined() ? packed.data_ptr<std::uint8_t>() : nullptr;
    auto* row_scales = scales.defined() ? scales.data_ptr<float>() : nullptr;
    if constexpr (N <= 16) {
        if (packed_rows) {
            const auto packed_blocks = static_cast<unsigned int>((rows - 1) / (block_threads / N) + 1);
            hadamard::packed_rows_kernel<T, N, Transform, Quantize><<<packed_blocks, block_threads, 0, stream>>>(
                source, destination, bytes, row_scales, rows, scale);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }
    if constexpr (N == 256 && Transform && Quantize) {
        if (contiguous256_fused) {
            hadamard::contiguous256_kernel<T, true, true><<<blocks, block_threads, 0, stream>>>(
                source, destination, bytes, row_scales, rows, scale);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }
    hadamard::launch_warp<T, N, Transform, Quantize>(
        source, destination, bytes, row_scales, rows, scale, block_threads, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T, bool Transform, bool Quantize>
void dispatch_dim(const at::Tensor& input, at::Tensor& output, at::Tensor& packed,
                  at::Tensor& scales, float scale, cudaStream_t stream, int block_threads,
                  bool contiguous256_fused, bool packed_rows) {
    switch (input.size(-1)) {
#define DIM_CASE(N) case N: launch<T, N, Transform, Quantize>(input, output, packed, scales, scale, stream, block_threads, contiguous256_fused, packed_rows); break
        DIM_CASE(1); DIM_CASE(2); DIM_CASE(4); DIM_CASE(8); DIM_CASE(16);
        DIM_CASE(32); DIM_CASE(64); DIM_CASE(128); DIM_CASE(256);
#undef DIM_CASE
        default: TORCH_CHECK(false, "unsupported dimension");
    }
}

template <bool Transform, bool Quantize>
void dispatch(const at::Tensor& input, at::Tensor& output, at::Tensor& packed,
              at::Tensor& scales, double scale, int block_threads, bool contiguous256_fused = false, bool packed_rows = false) {
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    if (input.scalar_type() == at::kHalf)
        dispatch_dim<__half, Transform, Quantize>(input, output, packed, scales, static_cast<float>(scale), stream, block_threads, contiguous256_fused, packed_rows);
    else
        dispatch_dim<__nv_bfloat16, Transform, Quantize>(input, output, packed, scales, static_cast<float>(scale), stream, block_threads, contiguous256_fused, packed_rows);
}

template <class T, int N>
void launch_pairs(const at::Tensor& input, at::Tensor& packed, at::Tensor& scales,
                  cudaStream_t stream, int block_threads) {
    const auto rows = static_cast<std::size_t>(input.numel() / N);
    const auto blocks = static_cast<unsigned int>((rows - 1) / (block_threads / (N / 2)) + 1);
    hadamard::quantize_pairs_kernel<T, N><<<blocks, block_threads, 0, stream>>>(
        reinterpret_cast<const T*>(input.data_ptr()), packed.data_ptr<std::uint8_t>(),
        scales.data_ptr<float>(), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T>
void dispatch_pairs(const at::Tensor& input, at::Tensor& packed, at::Tensor& scales,
                    cudaStream_t stream, int block_threads) {
    // Select the dimension before grid arithmetic so power-of-two divisions
    // stay compile-time specializations, matching the existing launch path.
    switch (input.size(-1)) {
#define PAIR_CASE(N) case N: launch_pairs<T, N>(input, packed, scales, stream, block_threads); break
        PAIR_CASE(2); PAIR_CASE(4); PAIR_CASE(8); PAIR_CASE(16);
#undef PAIR_CASE
        default: TORCH_CHECK(false, "paired quantization requires dimension 2, 4, 8, or 16");
    }
}

// Specialize only quantize-only dispatch. Hadamard launch/dispatch templates
// retain their existing definitions instead of inheriting the new routing body.
template <>
void dispatch<false, true>(const at::Tensor& input, at::Tensor& output, at::Tensor& packed,
                           at::Tensor& scales, double scale, int block_threads,
                           bool contiguous256_fused, bool packed_rows) {
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    if (packed_rows && input.size(-1) >= 2) {
        if (input.scalar_type() == at::kHalf)
            dispatch_pairs<__half>(input, packed, scales, stream, block_threads);
        else
            dispatch_pairs<__nv_bfloat16>(input, packed, scales, stream, block_threads);
        return;
    }
    if (input.scalar_type() == at::kHalf)
        dispatch_dim<__half, false, true>(input, output, packed, scales, static_cast<float>(scale), stream, block_threads, contiguous256_fused, packed_rows);
    else
        dispatch_dim<__nv_bfloat16, false, true>(input, output, packed, scales, static_cast<float>(scale), stream, block_threads, contiguous256_fused, packed_rows);
}

hadamard::RowChoice row_choice(const at::Tensor& input, const std::string& name,
                                  bool fused, int fallback_threads) {
    const auto layout = hadamard::parse_row_layout(name);
    TORCH_CHECK(layout != hadamard::RowLayout::Packed || input.size(-1) <= 16,
                "row_layout='packed' requires last dimension at most 16");
    auto device = hadamard::RowDevice::Unknown;
    if (layout == hadamard::RowLayout::Auto)
        device = hadamard::row_device(at::cuda::getDeviceProperties(input.get_device())->name);
    return hadamard::choose_rows(layout, device, input.numel() / input.size(-1),
                                  static_cast<int>(input.size(-1)), fused, fallback_threads);
}

at::Tensor transform(const at::Tensor& input, double scale, int block_threads,
                     const std::string& row_layout = "original") {
    validate(input, scale, block_threads);
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto output = at::empty_like(input);
    at::Tensor packed, scales;
    const auto choice = row_choice(input, row_layout, false, block_threads);
    dispatch<true, false>(input, output, packed, scales, scale, choice.threads, false, choice.packed);
    return output;
}

template <bool Transform>
std::tuple<at::Tensor, at::Tensor> quantized(const at::Tensor& input, double scale, int block_threads,
                                         const std::string& fused_layout = "original",
                                         const std::string& row_layout = "original") {
    validate(input, scale, block_threads);
    TORCH_CHECK(fused_layout == "original" || fused_layout == "contiguous256",
                "fused_layout must be 'original' or 'contiguous256'");
    TORCH_CHECK(fused_layout == "original" || row_layout == "original",
                "row_layout cannot be combined with contiguous256");
    const bool contiguous256_fused = fused_layout == "contiguous256";
    if (contiguous256_fused) {
        TORCH_CHECK(Transform, "contiguous256 is only supported for fused Hadamard INT4");
        TORCH_CHECK(input.size(-1) == 256, "fused_layout='contiguous256' requires last dimension 256");
        TORCH_CHECK(block_threads == 128, "fused_layout='contiguous256' requires block_threads=128");
    }
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto packed_shape = input.sizes().vec();
    packed_shape.back() = (packed_shape.back() + 1) / 2;
    auto scale_shape = input.sizes().vec();
    scale_shape.pop_back();
    auto packed = at::empty(packed_shape, input.options().dtype(at::kByte));
    auto scales = at::empty(scale_shape, input.options().dtype(at::kFloat));
    at::Tensor output;
    const auto choice = row_choice(input, row_layout, true, block_threads);
    dispatch<Transform, true>(input, output, packed, scales, scale, choice.threads, contiguous256_fused, choice.packed);
    return {packed, scales};
}

std::tuple<at::Tensor, at::Tensor> quantize_only(const at::Tensor& input, int block_threads) {
    return quantized<false>(input, 1.0, block_threads);
}

std::tuple<at::Tensor, at::Tensor> quantize_only_packed(const at::Tensor& input, int block_threads) {
    return quantized<false>(input, 1.0, block_threads, "original", "packed");
}

void validate_buffer(const at::Tensor& input, const at::Tensor& buffer,
                     at::ScalarType dtype, int rank, const char* name) {
    TORCH_CHECK(buffer.is_cuda() && buffer.device() == input.device(), name, " must be on the input CUDA device");
    TORCH_CHECK(buffer.scalar_type() == dtype, name, " has incorrect dtype");
    TORCH_CHECK(buffer.dim() == rank, name, " has incorrect rank");
    TORCH_CHECK(buffer.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!buffer.is_neg() && !buffer.is_conj(), name, " must not have lazy negative/conjugate metadata");
    TORCH_CHECK(!buffer.requires_grad(), name, " must not require gradients");
    TORCH_CHECK(!buffer.is_inference() || c10::InferenceMode::is_enabled(),
                name, " is an inference tensor; update it inside inference_mode");
}

void require_disjoint(const at::Tensor& first, const at::Tensor& second) {
    // Both tensors have validated contiguous shapes on one device. Compare the
    // active byte intervals, including distinct storage wrappers (e.g. DLPack).
    const auto a = reinterpret_cast<std::uintptr_t>(first.data_ptr());
    const auto b = reinterpret_cast<std::uintptr_t>(second.data_ptr());
    const auto a_size = static_cast<std::uintptr_t>(first.numel()) * first.element_size();
    const auto b_size = static_cast<std::uintptr_t>(second.numel()) * second.element_size();
    TORCH_CHECK(a < b ? b - a >= a_size : a - b >= b_size,
                "input and output buffers must have disjoint active byte ranges");
}

void transform_out(const at::Tensor& input, at::Tensor output, double scale,
                   int block_threads, const std::string& row_layout) {
    validate(input, scale, block_threads);
    validate_buffer(input, output, input.scalar_type(), input.dim(), "output");
    TORCH_CHECK(output.sizes() == input.sizes(), "output has incorrect shape");
    require_disjoint(input, output);
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto choice = row_choice(input, row_layout, false, block_threads);
    at::Tensor packed, scales;
    torch::autograd::increment_version(output);
    dispatch<true, false>(input, output, packed, scales, scale, choice.threads, false, choice.packed);
}

template <bool Transform>
void quantized_out(const at::Tensor& input, at::Tensor packed, at::Tensor scales,
                   double scale, int block_threads, const std::string& fused_layout,
                   const std::string& row_layout) {
    validate(input, scale, block_threads);
    TORCH_CHECK(fused_layout == "original" || fused_layout == "contiguous256",
                "fused_layout must be 'original' or 'contiguous256'");
    TORCH_CHECK(fused_layout == "original" || row_layout == "original",
                "row_layout cannot be combined with contiguous256");
    const bool contiguous256 = fused_layout == "contiguous256";
    if (contiguous256) {
        TORCH_CHECK(Transform && input.size(-1) == 256 && block_threads == 128,
                    "contiguous256 requires fused Hadamard, dimension 256, and 128 threads");
    }
    validate_buffer(input, packed, at::kByte, input.dim(), "packed output");
    validate_buffer(input, scales, at::kFloat, input.dim() - 1, "scales output");
    TORCH_CHECK(packed.size(-1) == (input.size(-1) + 1) / 2, "packed output has incorrect shape");
    for (int i = 0; i < input.dim() - 1; ++i) {
        TORCH_CHECK(packed.size(i) == input.size(i), "packed output has incorrect shape");
        TORCH_CHECK(scales.size(i) == input.size(i), "scales output has incorrect shape");
    }
    require_disjoint(input, packed);
    require_disjoint(input, scales);
    require_disjoint(packed, scales);
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto choice = row_choice(input, row_layout, true, block_threads);
    at::Tensor output;
    torch::autograd::increment_version(packed);
    torch::autograd::increment_version(scales);
    dispatch<Transform, true>(input, output, packed, scales, scale, choice.threads, contiguous256, choice.packed);
}

void quantize_only_out(const at::Tensor& input, at::Tensor packed, at::Tensor scales, int block_threads) {
    quantized_out<false>(input, packed, scales, 1.0, block_threads, "original", "original");
}

void quantize_only_packed_out(const at::Tensor& input, at::Tensor packed, at::Tensor scales, int block_threads) {
    quantized_out<false>(input, packed, scales, 1.0, block_threads, "original", "packed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("hadamard", &transform, pybind11::arg("input"), pybind11::arg("scale") = 1.0, pybind11::arg("block_threads") = 128, pybind11::arg("row_layout") = "original",
               "Forward-only last-axis Hadamard; finite CUDA FP16/BF16 input, 2D or 4D contiguous. block_threads=128 (default) or 256; row_layout=original (default), packed (N<=16), or auto (verified model/range rules; may select 256 threads).");
    module.def("hadamard_int4", &quantized<true>, pybind11::arg("input"), pybind11::arg("scale") = 1.0, pybind11::arg("block_threads") = 128,
               pybind11::arg("fused_layout") = "original", pybind11::arg("row_layout") = "original",
               "Fused transform and rowwise symmetric INT4; returns (uint8 packed, float32 scales). fused_layout='original' (default) supports block_threads=128 or 256; explicit 'contiguous256' requires N256 and block_threads=128. row_layout=packed/auto selects small-N row packing and cannot be combined with contiguous256.");
    module.def("quantize_int4", &quantize_only, pybind11::arg("input"), pybind11::arg("block_threads") = 128,
               "Quantize an already-rounded FP16/BF16 tensor; even values occupy the low nibble. block_threads=128 (default) or 256.");
    module.def("quantize_int4_packed", &quantize_only_packed, pybind11::arg("input"), pybind11::arg("block_threads") = 128,
               "Opt-in row-packed INT4 quantization for N<=16. Same rounding and packed output contract as quantize_int4; no automatic device policy.");
    module.def("hadamard_out", &transform_out, pybind11::arg("input"), pybind11::arg("output"),
               pybind11::arg("scale") = 1.0, pybind11::arg("block_threads") = 128, pybind11::arg("row_layout") = "original",
               "Write Hadamard into a preallocated, disjoint same-shape/dtype output; returns None. No resizing or output allocation.");
    module.def("hadamard_int4_out", &quantized_out<true>, pybind11::arg("input"), pybind11::arg("packed"), pybind11::arg("scales"),
               pybind11::arg("scale") = 1.0, pybind11::arg("block_threads") = 128,
               pybind11::arg("fused_layout") = "original", pybind11::arg("row_layout") = "original",
               "Write fused INT4 into preallocated uint8 packed and float32 scales buffers; returns None. All active byte ranges must be disjoint.");
    module.def("quantize_int4_out", &quantize_only_out, pybind11::arg("input"), pybind11::arg("packed"), pybind11::arg("scales"),
               pybind11::arg("block_threads") = 128,
               "Write INT4 quantization into preallocated disjoint buffers; returns None. No resizing or output allocation.");
    module.def("quantize_int4_packed_out", &quantize_only_packed_out, pybind11::arg("input"), pybind11::arg("packed"), pybind11::arg("scales"),
               pybind11::arg("block_threads") = 128,
               "Write row-packed INT4 quantization for N<=16 into disjoint caller buffers. Returns None; no resizing or output allocation.");
}
