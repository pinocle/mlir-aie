//===- add.cc -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef ADD_ELEMS
#define ADD_ELEMS size
#define ADD_ELEMS_RUNTIME
#endif

template <typename T_in, typename T_out, const int N>
void eltwise_add(T_in *a, T_in *b, T_out *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

// The untuned path runs four independent load/add/store chains per iteration.
// One aie::add chain is latency-bound: both operand loads are vlda.conv, which
// only the a port has.
#define ADD_UNROLL 4

// Lanes the vector loops walk: bf16 where the vector unit has it; on an
// architecture without bf16, one vector of AIE_F32_LANES x 32 bits of the
// element type (AIE1: 8 f32/i32, 16 i16, 32 i8; see aie_arch.h).
template <typename T> constexpr int eltwise_add_lanes() {
  return std::is_same_v<T, bfloat16> ? AIE_BF16_LANES
                                     : AIE_F32_LANES * 4 / (int)sizeof(T);
}

// The tuned paths run one chain per iteration, which pipelines (II1 on AIE2P)
// with restrict pointers and a rolled loop. Only a converts on load; b loads
// as bf16 on the b port and is added as b * 1 in a mac.
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
template <typename T_in, typename T_out, int vec_factor>
void eltwise_vadd_mac(aie::restrict_vector_iterator<T_in, vec_factor> &pA,
                      aie::restrict_vector_iterator<T_in, vec_factor> &pB,
                      aie::restrict_vector_iterator<T_out, vec_factor> &pC,
                      int n) {
  const auto ones = aie::broadcast<T_in, vec_factor>(1.0f);
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < n; i++) {
    aie::accum<accfloat, vec_factor> acc;
    acc.from_vector(*pA++);
    *pC++ = aie::mac(acc, *pB++, ones).template to_vector<T_out>();
  }
}
#endif

template <typename T_in, typename T_out, const int N>
void eltwise_vadd(T_in *__restrict a, T_in *__restrict b, T_out *__restrict c) {

  constexpr int vec_factor = eltwise_add_lanes<T_in>();
  event0();
  auto pA1 = aie::begin_restrict_vector<vec_factor>(a);
  auto pB1 = aie::begin_restrict_vector<vec_factor>(b);
  auto pC1 = aie::begin_restrict_vector<vec_factor>(c);
  constexpr int F = N / vec_factor;
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  eltwise_vadd_mac<T_in, T_out, vec_factor>(pA1, pB1, pC1,
                                            F / ADD_UNROLL * ADD_UNROLL);
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < F / ADD_UNROLL; i++) {
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    auto A1 = *pA1++;
    auto B1 = *pB1++;
    auto A2 = *pA1++;
    auto B2 = *pB1++;
    auto A3 = *pA1++;
    auto B3 = *pB1++;
    *pC1++ = aie::add(A0, B0);
    *pC1++ = aie::add(A1, B1);
    *pC1++ = aie::add(A2, B2);
    *pC1++ = aie::add(A3, B3);
  }
#endif
  // Whole vectors past the last full group of four.
  for (int i = 0; i < F % ADD_UNROLL; i++) {
    *pC1++ = aie::add(*pA1++, *pB1++);
  }
  event1();
}

// Runtime size (need not divide vec_factor); scalar tail avoids the full-width
// load_v/store_v reading/writing past the buffer on a short final vector.
template <typename T_in, typename T_out>
void eltwise_vadd_size(T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c, int size) {
  constexpr int vec_factor = eltwise_add_lanes<T_in>();
  event0();
  auto pA1 = aie::begin_restrict_vector<vec_factor>(a);
  auto pB1 = aie::begin_restrict_vector<vec_factor>(b);
  auto pC1 = aie::begin_restrict_vector<vec_factor>(c);
  // Unsigned, so F % ADD_UNROLL is a mask rather than a __modsi3 call.
  const int F = (uint32_t)ADD_ELEMS / vec_factor;
// The single chain pipelines only with a compile-time trip count.
#if (AIE_TUNED_AIE2 || AIE_TUNED_AIE2P) && !defined(ADD_ELEMS_RUNTIME)
  eltwise_vadd_mac<T_in, T_out, vec_factor>(pA1, pB1, pC1,
                                            F / ADD_UNROLL * ADD_UNROLL);
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < F / ADD_UNROLL; i++) { // see eltwise_vadd
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    auto A1 = *pA1++;
    auto B1 = *pB1++;
    auto A2 = *pA1++;
    auto B2 = *pB1++;
    auto A3 = *pA1++;
    auto B3 = *pB1++;
    *pC1++ = aie::add(A0, B0);
    *pC1++ = aie::add(A1, B1);
    *pC1++ = aie::add(A2, B2);
    *pC1++ = aie::add(A3, B3);
  }
#endif
  if ((uint32_t)ADD_ELEMS % (vec_factor * ADD_UNROLL)) {
    for (int i = 0; i < F % ADD_UNROLL; i++) {
      *pC1++ = aie::add(*pA1++, *pB1++);
    }
    // Scalar tail for a size that is not a whole number of vectors.
    const int done = F * vec_factor;
    const int tail = ADD_ELEMS - done;
    for (int i = 0; i < tail; i++) {
      c[done + i] = a[done + i] + b[done + i];
    }
  }
  event1();
}

// AIE_KERNELS_TEMPLATES_ONLY: a unit that instantiates the templates above
// at its own sizes leaves out these entry points, which would otherwise all
// be linked into a core with 16 KiB of program memory.
#ifndef AIE_KERNELS_TEMPLATES_ONLY
extern "C" {

#if AIE_BF16_LANES
void eltwise_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_add<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_vadd<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_add_bf16_vector_size(bfloat16 *a_in, bfloat16 *b_in,
                                  bfloat16 *c_out, int size) {
  eltwise_vadd_size<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}
#endif // AIE_BF16_LANES

#if AIE_F32_LANES
void eltwise_add_f32_vector(float *a_in, float *b_in, float *c_out) {
  eltwise_vadd<float, float, 1024>(a_in, b_in, c_out);
}

void eltwise_add_f32_vector_size(float *a_in, float *b_in, float *c_out,
                                  int size) {
  eltwise_vadd_size<float, float>(a_in, b_in, c_out, size);
}

void eltwise_add_i32_vector(int32_t *a_in, int32_t *b_in, int32_t *c_out) {
  eltwise_vadd<int32_t, int32_t, 1024>(a_in, b_in, c_out);
}

void eltwise_add_i32_vector_size(int32_t *a_in, int32_t *b_in, int32_t *c_out,
                                  int size) {
  eltwise_vadd_size<int32_t, int32_t>(a_in, b_in, c_out, size);
}

void eltwise_add_i16_vector(int16_t *a_in, int16_t *b_in, int16_t *c_out) {
  eltwise_vadd<int16_t, int16_t, 1024>(a_in, b_in, c_out);
}

void eltwise_add_i16_vector_size(int16_t *a_in, int16_t *b_in, int16_t *c_out,
                                  int size) {
  eltwise_vadd_size<int16_t, int16_t>(a_in, b_in, c_out, size);
}

void eltwise_add_i8_vector(int8_t *a_in, int8_t *b_in, int8_t *c_out) {
  eltwise_vadd<int8_t, int8_t, 1024>(a_in, b_in, c_out);
}

void eltwise_add_i8_vector_size(int8_t *a_in, int8_t *b_in, int8_t *c_out,
                                  int size) {
  eltwise_vadd_size<int8_t, int8_t>(a_in, b_in, c_out, size);
}
#endif // AIE_F32_LANES

} // extern "C"
#endif // AIE_KERNELS_TEMPLATES_ONLY
