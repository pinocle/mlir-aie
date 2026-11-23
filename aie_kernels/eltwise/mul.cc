//===- mul.cc -------------------------------------------------*- C++ -*-===//
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

#ifndef MUL_ELEMS
#define MUL_ELEMS size
#define MUL_ELEMS_RUNTIME
#endif

template <typename T_in, typename T_out, const int N>
void eltwise_mul(T_in *a, T_in *b, T_out *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
  }
}

// The untuned path runs four independent load/mul/store chains per iteration;
// see ADD_UNROLL in add.cc.
#define MUL_UNROLL 4

// Lanes the vector loops walk: bf16 where the vector unit has it; on an
// architecture without bf16, one vector of AIE_F32_LANES x 32 bits of the
// element type (AIE1: 8 f32/i32, 16 i16, 32 i8; see aie_arch.h).
template <typename T> constexpr int eltwise_mul_lanes() {
  return std::is_same_v<T, bfloat16> ? AIE_BF16_LANES
                                     : AIE_F32_LANES * 4 / (int)sizeof(T);
}

// aie::mul on bf16 yields an accumulator (fp32 products); convert back to
// T_out explicitly.  Assigning the accumulator straight into a vector<T_out>
// produces garbage at the 32-wide AIE2P width.
#define MUL_ONE(A, B) (aie::mul((A), (B)).template to_vector<T_out>())

// The tuned paths run one chain per iteration instead, which pipelines (II1
// on AIE2P) with restrict pointers and a rolled loop.
template <typename T_in, typename T_out, const int N>
void eltwise_vmul(T_in *__restrict a, T_in *__restrict b, T_out *__restrict c) {

  constexpr int vec_factor = eltwise_mul_lanes<T_in>();
  event0();
  auto pA1 = aie::begin_restrict_vector<vec_factor>(a);
  auto pB1 = aie::begin_restrict_vector<vec_factor>(b);
  auto pC1 = aie::begin_restrict_vector<vec_factor>(c);
  constexpr int F = N / vec_factor;
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < F / MUL_UNROLL * MUL_UNROLL; i++) {
    *pC1++ = MUL_ONE(*pA1++, *pB1++);
  }
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < F / MUL_UNROLL; i++) {
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    auto A1 = *pA1++;
    auto B1 = *pB1++;
    auto A2 = *pA1++;
    auto B2 = *pB1++;
    auto A3 = *pA1++;
    auto B3 = *pB1++;
    *pC1++ = MUL_ONE(A0, B0);
    *pC1++ = MUL_ONE(A1, B1);
    *pC1++ = MUL_ONE(A2, B2);
    *pC1++ = MUL_ONE(A3, B3);
  }
#endif
  // Whole vectors past the last full group of four.
  for (int i = 0; i < F % MUL_UNROLL; i++) {
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    *pC1++ = MUL_ONE(A0, B0);
  }
  event1();
}

// Runtime size (need not divide vec_factor); scalar tail avoids the full-width
// load_v/store_v reading/writing past the buffer on a short final vector.
template <typename T_in, typename T_out>
void eltwise_vmul_size(T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c, int size) {
  constexpr int vec_factor = eltwise_mul_lanes<T_in>();
  event0();
  auto pA1 = aie::begin_restrict_vector<vec_factor>(a);
  auto pB1 = aie::begin_restrict_vector<vec_factor>(b);
  auto pC1 = aie::begin_restrict_vector<vec_factor>(c);
  const int F = (uint32_t)MUL_ELEMS / vec_factor; // see eltwise_vadd_size
// The single chain pipelines only with a compile-time trip count.
#if (AIE_TUNED_AIE2 || AIE_TUNED_AIE2P) && !defined(MUL_ELEMS_RUNTIME)
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < F / MUL_UNROLL * MUL_UNROLL; i++) {
    *pC1++ = MUL_ONE(*pA1++, *pB1++);
  }
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < F / MUL_UNROLL; i++) { // see eltwise_vmul
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    auto A1 = *pA1++;
    auto B1 = *pB1++;
    auto A2 = *pA1++;
    auto B2 = *pB1++;
    auto A3 = *pA1++;
    auto B3 = *pB1++;
    *pC1++ = MUL_ONE(A0, B0);
    *pC1++ = MUL_ONE(A1, B1);
    *pC1++ = MUL_ONE(A2, B2);
    *pC1++ = MUL_ONE(A3, B3);
  }
#endif
  if ((uint32_t)MUL_ELEMS % (vec_factor * MUL_UNROLL)) {
    for (int i = 0; i < F % MUL_UNROLL; i++) {
      auto A0 = *pA1++;
      auto B0 = *pB1++;
      *pC1++ = MUL_ONE(A0, B0);
    }
    // Scalar tail for a size that is not a whole number of vectors.
    const int done = F * vec_factor;
    const int tail = MUL_ELEMS - done;
    for (int i = 0; i < tail; i++) {
      c[done + i] = a[done + i] * b[done + i];
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
void eltwise_mul_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_mul<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_mul_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_vmul<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_mul_bf16_vector_size(bfloat16 *a_in, bfloat16 *b_in,
                                  bfloat16 *c_out, int size) {
  eltwise_vmul_size<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}
#endif // AIE_BF16_LANES

#if AIE_F32_LANES
void eltwise_mul_f32_vector(float *a_in, float *b_in, float *c_out) {
  eltwise_vmul<float, float, 1024>(a_in, b_in, c_out);
}

void eltwise_mul_f32_vector_size(float *a_in, float *b_in, float *c_out,
                                  int size) {
  eltwise_vmul_size<float, float>(a_in, b_in, c_out, size);
}

void eltwise_mul_i32_vector(int32_t *a_in, int32_t *b_in, int32_t *c_out) {
  eltwise_vmul<int32_t, int32_t, 1024>(a_in, b_in, c_out);
}

void eltwise_mul_i32_vector_size(int32_t *a_in, int32_t *b_in, int32_t *c_out,
                                  int size) {
  eltwise_vmul_size<int32_t, int32_t>(a_in, b_in, c_out, size);
}

void eltwise_mul_i16_vector(int16_t *a_in, int16_t *b_in, int16_t *c_out) {
  eltwise_vmul<int16_t, int16_t, 1024>(a_in, b_in, c_out);
}

void eltwise_mul_i16_vector_size(int16_t *a_in, int16_t *b_in, int16_t *c_out,
                                  int size) {
  eltwise_vmul_size<int16_t, int16_t>(a_in, b_in, c_out, size);
}

void eltwise_mul_i8_vector(int8_t *a_in, int8_t *b_in, int8_t *c_out) {
  eltwise_vmul<int8_t, int8_t, 1024>(a_in, b_in, c_out);
}

void eltwise_mul_i8_vector_size(int8_t *a_in, int8_t *b_in, int8_t *c_out,
                                  int size) {
  eltwise_vmul_size<int8_t, int8_t>(a_in, b_in, c_out, size);
}
#endif // AIE_F32_LANES

} // extern "C"
#endif // AIE_KERNELS_TEMPLATES_ONLY
