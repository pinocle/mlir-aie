//===- relu_aie1.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE1 has no bf16: the same loop as relu_aie2p.h on f32, AIE_F32_LANES at a
// time (one 256-bit AIE1 vector register).

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef RELU_ELEMS
#define RELU_ELEMS vector_size
#endif

void relu_vectorized_f32(float *restrict a, float *restrict c,
                         const int32_t vector_size) {
  event0();

  auto it_in = aie::begin_restrict_vector<AIE_F32_LANES>(a);
  auto it_out = aie::begin_restrict_vector<AIE_F32_LANES>(c);

  vector<float, AIE_F32_LANES> zeroes = aie::zeros<float, AIE_F32_LANES>();

  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < RELU_ELEMS; i += AIE_F32_LANES) {
    vector<float, AIE_F32_LANES> input = *it_in++;
    *it_out++ = aie::max(input, zeroes);
  }

  event1();
}

extern "C" {

void f32_relu(float *a_in, float *c_out) { relu_vectorized_f32(a_in, c_out, 1024); }

void relu_f32_size(float *restrict input, float *restrict output,
                   int32_t input_size) {
  relu_vectorized_f32(input, output, input_size);
}

} // extern "C"
