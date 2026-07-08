//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

// NOTE(HIP/AMD): <cuda_bf16.h>/<cuda_fp16.h> are NVIDIA-only headers; use the
// platform-portable wrapper that selects <hip/hip_bf16.h>/<hip/hip_fp16.h> on HIP.
#include <cuda/std/__floating_point/cuda_fp_types.h>

#include <cuda/__complex_>
#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC void test_is_trivially_copyable()
{
  static_assert(cuda::is_trivially_copyable<T>::value);
  static_assert(cuda::is_trivially_copyable<const T>::value);
  static_assert(cuda::is_trivially_copyable_v<T>);
  static_assert(cuda::is_trivially_copyable_v<const T>);
}

template <class T>
struct TriviallyCopyableWrapper
{
  T x;
};

struct TrivialPod
{
  int x;
  float y;
};

class NonTriviallyCopyable
{
public:
  TEST_FUNC NonTriviallyCopyable(const NonTriviallyCopyable&) {} // NOLINT
};

template <class T>
TEST_FUNC void test_is_trivially_copyable_compositions()
{
  test_is_trivially_copyable<T[4]>();
  test_is_trivially_copyable<cuda::std::array<T, 4>>();
  test_is_trivially_copyable<cuda::std::pair<T, T>>();
  test_is_trivially_copyable<cuda::std::tuple<T, T>>();
  test_is_trivially_copyable<cuda::std::complex<T>>();
  test_is_trivially_copyable<cuda::complex<T>>();
  test_is_trivially_copyable<TriviallyCopyableWrapper<T>>();
}

TEST_FUNC void test_composite_types()
{
  test_is_trivially_copyable<int[4]>();

  test_is_trivially_copyable<TrivialPod>();
  test_is_trivially_copyable<TrivialPod[2]>();

  // cuda::std::array, pair, tuple, complex, and aggregate wrappers of trivially copyable types
  test_is_trivially_copyable_compositions<int>();
  test_is_trivially_copyable_compositions<float>();
  test_is_trivially_copyable<cuda::std::tuple<>>();

  // non-trivially copyable types
  static_assert(!cuda::is_trivially_copyable_v<NonTriviallyCopyable>);
}

TEST_FUNC void test_extended_fp_types()
{
#if _CCCL_HAS_NVFP16()
  test_is_trivially_copyable_compositions<__half>();
  test_is_trivially_copyable_compositions<__half2>();
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  test_is_trivially_copyable_compositions<__nv_bfloat16>();
  test_is_trivially_copyable_compositions<__nv_bfloat162>();
#endif // _CCCL_HAS_NVBF16()
}

int main(int, char**)
{
  test_composite_types();
  test_extended_fp_types();
  return 0;
}
