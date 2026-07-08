//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

// <cuda/std/complex>

// template<class T>
// class complex
// {
// public:
//   using value_type = T;
//   ...
// };

#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC void test()
{
  using C = cuda::std::complex<T>;

  static_assert(cuda::std::is_default_constructible_v<C>);
  static_assert(!cuda::std::is_trivially_default_constructible_v<C>);
  static_assert(cuda::std::is_nothrow_default_constructible_v<C>);

  // NOTE(HIP/AMD): On HIP, __half and __nv_bfloat16 are trivially copyable POD structs
  // (unlike NVIDIA CUDA where they have user-defined copy operations), so complex<T> is
  // trivially copyable whenever T is trivially copyable.  Using is_trivially_copyable_v<T>
  // instead of is_floating_point_v<T> produces correct results for both NVIDIA and HIP:
  // standard fp types (float/double/long double) are always trivially copyable, and
  // NVIDIA __half / __nv_bfloat16 are not trivially copyable, matching is_floating_point_v.
  static_assert(cuda::std::is_copy_constructible_v<C>);
  static_assert(cuda::std::is_trivially_copy_constructible_v<C> == cuda::std::is_trivially_copyable_v<T>);
  static_assert(cuda::std::is_nothrow_copy_constructible_v<C>);

  static_assert(cuda::std::is_move_constructible_v<C>);
  static_assert(cuda::std::is_trivially_move_constructible_v<C> == cuda::std::is_trivially_copyable_v<T>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<C>);

  static_assert(cuda::std::is_copy_assignable_v<C>);
  static_assert(cuda::std::is_trivially_copy_assignable_v<C> == cuda::std::is_trivially_copyable_v<T>);
  static_assert(cuda::std::is_nothrow_copy_assignable_v<C>);

  static_assert(cuda::std::is_move_assignable_v<C>);
  static_assert(cuda::std::is_trivially_move_assignable_v<C> == cuda::std::is_trivially_copyable_v<T>);
  static_assert(cuda::std::is_nothrow_move_assignable_v<C>);

  static_assert(cuda::std::is_trivially_destructible_v<C>);
  static_assert(cuda::std::is_trivially_copyable_v<C> == cuda::std::is_trivially_copyable_v<T>);
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if !TEST_COMPILER(GCC, <, 10) // Old GCC considers the defaulted constructors as deleted
#  if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#  endif // _LIBCUDACXX_HAS_NVFP16()
#  if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#  endif // _LIBCUDACXX_HAS_NVBF16()
#endif // !TEST_COMPILER(GCC, <, 01)

  return 0;
}
