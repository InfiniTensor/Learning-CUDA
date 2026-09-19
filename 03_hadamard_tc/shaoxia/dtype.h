#pragma once

// Shared dtype tag used by hadacore.cu (kernel definition) and bench.cu (host
// caller). Must live in a header so both TUs see the *same* enum type.
enum class DType : int { Half, BFloat16 };
