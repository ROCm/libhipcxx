//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cuda/std/__simd_> // IWYU pragma: keep
#include <cuda/std/array>

namespace simd = cuda::std::simd;

using Vec_f32_4 = simd::basic_vec<float, simd::fixed_size<4>>;

extern "C" __global__ void test_operator_plus_f32_4(const float* lhs, const float* rhs, float* out)
{
  const cuda::std::array<float, 4> lhs_values{lhs[0], lhs[1], lhs[2], lhs[3]};
  const cuda::std::array<float, 4> rhs_values{rhs[0], rhs[1], rhs[2], rhs[3]};

  const Vec_f32_4 lhs_vec(lhs_values);
  const Vec_f32_4 rhs_vec(rhs_values);
  const Vec_f32_4 result = lhs_vec + rhs_vec;

  out[0] = result[0];
  out[1] = result[1];
  out[2] = result[2];
  out[3] = result[3];
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : test_operator_plus_f32_4
; SM100: {{.*FADD2.*}}
; SM100: {{.*FADD2.*}}

*/

/*
; ----- AMD/HIP additions (checked on HIP: clang -S AMDGCN ISA, gfx90a) -----
; HIP_ISA-LABEL: test_operator_plus_f32_4:
; HIP_ISA: v_pk_add_f32
; HIP_ISA: v_pk_add_f32
*/
