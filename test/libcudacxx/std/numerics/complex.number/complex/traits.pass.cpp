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

  // NOTE(HIP/AMD): what upstream is really asserting here is "complex<T> is trivial iff its
  // storage is", and it spells that as is_floating_point_v<T> because on CUDA the two agree:
  // complex<__half> / complex<__nv_bfloat16> store a __half2 / __nv_bfloat162, and those
  // vector types carry user-provided copy operations, so they are non-trivial exactly where
  // is_floating_point_v<T> is false.  (Note it is the *vector* type that is non-trivial --
  // CUDA's scalar __half has no user-provided copy operations and is trivially copyable, so
  // is_trivially_copyable_v<T> is NOT a drop-in for is_floating_point_v<T> on CUDA.)
  //
  // On HIP the coincidence breaks: __half2 is a trivially copyable POD, and complex<
  // __nv_bfloat16> uses the trivially copyable stand-in added in <cuda/std/__complex/nvbf16.h>
  // precisely so the storage stays trivial.  So complex<T> here is trivial for all four T,
  // while is_floating_point_v<T> is still false for the two extended types.
  //
  // Diverge only on the HIP path, so the CUDA expectation stays exactly upstream's.
#if _CCCL_HIP_COMPILATION()
  constexpr bool expected_trivial = cuda::std::is_trivially_copyable_v<T>;
#else // ^^^ HIP ^^^ / vvv CUDA vvv
  constexpr bool expected_trivial = cuda::std::is_floating_point_v<T>;
#endif // ^^^ CUDA ^^^

  static_assert(cuda::std::is_copy_constructible_v<C>);
  static_assert(cuda::std::is_trivially_copy_constructible_v<C> == expected_trivial);
  static_assert(cuda::std::is_nothrow_copy_constructible_v<C>);

  static_assert(cuda::std::is_move_constructible_v<C>);
  static_assert(cuda::std::is_trivially_move_constructible_v<C> == expected_trivial);
  static_assert(cuda::std::is_nothrow_move_constructible_v<C>);

  static_assert(cuda::std::is_copy_assignable_v<C>);
  static_assert(cuda::std::is_trivially_copy_assignable_v<C> == expected_trivial);
  static_assert(cuda::std::is_nothrow_copy_assignable_v<C>);

  static_assert(cuda::std::is_move_assignable_v<C>);
  static_assert(cuda::std::is_trivially_move_assignable_v<C> == expected_trivial);
  static_assert(cuda::std::is_nothrow_move_assignable_v<C>);

  static_assert(cuda::std::is_trivially_destructible_v<C>);
  static_assert(cuda::std::is_trivially_copyable_v<C> == expected_trivial);
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
