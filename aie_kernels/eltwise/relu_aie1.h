//===- relu_aie1.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE1 has no bf16: the same loop as relu_aie2p.h on f32 and on integers.

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef RELU_ELEMS
#define RELU_ELEMS vector_size
#endif

// One 256-bit AIE1 vector of the element type: 8 f32/i32, 16 i16, 32 i8.
template <typename T>
void relu_vectorized_aie1(T *restrict a, T *restrict c,
                          const int32_t vector_size) {
  constexpr int L = AIE_F32_LANES * 4 / (int)sizeof(T);
  event0();

  auto it_in = aie::begin_restrict_vector<L>(a);
  auto it_out = aie::begin_restrict_vector<L>(c);

  vector<T, L> zeroes = aie::zeros<T, L>();

  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < RELU_ELEMS; i += L) {
    vector<T, L> input = *it_in++;
    *it_out++ = aie::max(input, zeroes);
  }

  event1();
}

extern "C" {

#if !AIE_ONE_TYPE || defined(f32_ONLY)
void f32_relu(float *a_in, float *c_out) {
  relu_vectorized_aie1<float>(a_in, c_out, 1024);
}

void relu_f32_size(float *restrict input, float *restrict output,
                   int32_t input_size) {
  relu_vectorized_aie1<float>(input, output, input_size);
}
#endif // f32

#if !AIE_ONE_TYPE || defined(i32_ONLY)
void i32_relu(int32_t *a_in, int32_t *c_out) {
  relu_vectorized_aie1<int32_t>(a_in, c_out, 1024);
}

void relu_i32_size(int32_t *restrict input, int32_t *restrict output,
                   int32_t input_size) {
  relu_vectorized_aie1<int32_t>(input, output, input_size);
}
#endif // i32

#if !AIE_ONE_TYPE || defined(i16_ONLY)
void i16_relu(int16_t *a_in, int16_t *c_out) {
  relu_vectorized_aie1<int16_t>(a_in, c_out, 1024);
}

void relu_i16_size(int16_t *restrict input, int16_t *restrict output,
                   int32_t input_size) {
  relu_vectorized_aie1<int16_t>(input, output, input_size);
}
#endif // i16

#if !AIE_ONE_TYPE || defined(i8_ONLY)
void i8_relu(int8_t *a_in, int8_t *c_out) {
  relu_vectorized_aie1<int8_t>(a_in, c_out, 1024);
}

void relu_i8_size(int8_t *restrict input, int8_t *restrict output,
                  int32_t input_size) {
  relu_vectorized_aie1<int8_t>(input, output, input_size);
}
#endif // i8

} // extern "C"
