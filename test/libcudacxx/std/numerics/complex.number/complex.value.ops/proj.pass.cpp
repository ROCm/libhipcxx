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

// TODO(HIP/AMD): hipcc compiler issue with complex transcendentals - ticket ROCM-20557
// UNSUPPORTED: hipcc

// <complex>

// template<class T>
//   complex<T>
//   proj(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(const cuda::std::complex<T>& z, cuda::std::complex<T> x)
{
  assert(proj(z) == x);
}

template <class T>
__host__ __device__ void test()
{
  test(cuda::std::complex<T>(1, 2), cuda::std::complex<T>(1, 2));
  test(cuda::std::complex<T>(-1, 2), cuda::std::complex<T>(-1, 2));
  test(cuda::std::complex<T>(1, -2), cuda::std::complex<T>(1, -2));
  test(cuda::std::complex<T>(-1, -2), cuda::std::complex<T>(-1, -2));
}

template <class T>
__host__ __device__ void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    cuda::std::complex<T> r = proj(testcases[i]);
    switch (classify(testcases[i]))
    {
      case zero:
      case non_zero:
        assert(r == testcases[i]);
        assert(cuda::std::signbit(real(r)) == cuda::std::signbit(real(testcases[i])));
        assert(cuda::std::signbit(imag(r)) == cuda::std::signbit(imag(testcases[i])));
        break;
      case inf:
        assert(cuda::std::isinf(real(r)) && real(r) > T(0));
        assert(imag(r) == T(0));
        assert(cuda::std::signbit(imag(r)) == cuda::std::signbit(imag(testcases[i])));
        break;
      case NaN:
      case non_zero_nan:
        assert(classify(r) == classify(testcases[i]));
        break;
    }
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  test_edges<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test_edges<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_edges<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}
