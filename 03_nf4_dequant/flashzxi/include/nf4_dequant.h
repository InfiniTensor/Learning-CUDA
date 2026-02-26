//
// Created by core_dump on 2/25/26.
//

#pragma once

#include "quant_state.h"
void nf4_dequant_naive(const QuantState& quant_state, __half* output);
void nf4_dequant_warp8_batch32_two_phase(const QuantState& quant_state, __half* output);
void nf4_dequant_warp8_batch32_one_phase(const QuantState& quant_state, __half* output);