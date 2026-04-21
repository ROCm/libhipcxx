//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

_CCCL_SUPPRESS_DEPRECATED_PUSH

template <class VType, size_t Size>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::tuple_size<VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<const VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<volatile VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<const volatile VType>::value == Size, "");
}

#define EXPAND_VECTOR_TYPE(Type) \
  test<Type##1, 1>();            \
  test<Type##2, 2>();            \
  test<Type##3, 3>();            \
  test<Type##4, 4>();

__host__ __device__ constexpr bool test()
{
  EXPAND_VECTOR_TYPE(char);
  EXPAND_VECTOR_TYPE(uchar);
  EXPAND_VECTOR_TYPE(short);
  EXPAND_VECTOR_TYPE(ushort);
  EXPAND_VECTOR_TYPE(int);
  EXPAND_VECTOR_TYPE(uint);
  EXPAND_VECTOR_TYPE(long);
  EXPAND_VECTOR_TYPE(ulong);
  EXPAND_VECTOR_TYPE(longlong);
  EXPAND_VECTOR_TYPE(ulonglong);
  EXPAND_VECTOR_TYPE(float);
  EXPAND_VECTOR_TYPE(double);

#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long4_16a, 4>();
  test<long4_32a, 4>();
  test<ulong4_16a, 4>();
  test<ulong4_32a, 4>();
  test<longlong4_16a, 4>();
  test<longlong4_32a, 4>();
  test<ulonglong4_16a, 4>();
  test<ulonglong4_32a, 4>();
  test<double4_16a, 4>();
  test<double4_32a, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

#if _CCCL_HAS_NVFP16()
  test<__half2, 2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat162, 2>();
#endif // _CCCL_HAS_NVBF16()

  test<dim3, 3>();

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test());
  return 0;
}
