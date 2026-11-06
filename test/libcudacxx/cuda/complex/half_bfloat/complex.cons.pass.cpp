//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
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

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T, class U>
TEST_FUNC void test_construction()
{
  const cuda::std::complex<U> only_real{static_cast<U>(42.0), static_cast<U>(0.0)};
  const cuda::std::complex<U> only_imag{static_cast<U>(0.0), static_cast<U>(42.0)};
  const cuda::std::complex<U> real_imag{static_cast<U>(42.0), static_cast<U>(112.0)};

  const cuda::std::complex<T> from_only_real{only_real};
  const cuda::std::complex<T> from_only_imag{only_imag};
  const cuda::std::complex<T> from_real_imag{real_imag};

  assert(from_only_real.real() == static_cast<T>(42.0));
  assert(from_only_real.imag() == static_cast<T>(0.0));
  assert(from_only_imag.real() == static_cast<T>(0.0));
  assert(from_only_imag.imag() == static_cast<T>(42.0));
  assert(from_real_imag.real() == static_cast<T>(42.0));
  assert(from_real_imag.imag() == static_cast<T>(112.0));
}

TEST_FUNC void test()
{
#if _LIBCUDACXX_HAS_NVFP16()
  test_construction<__half, float>();
  test_construction<__half, double>();
  test_construction<float, __half>();
  test_construction<double, __half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_construction<__hip_bfloat16, float>();
  test_construction<__hip_bfloat16, double>();
  test_construction<float, __hip_bfloat16>();
  test_construction<double, __hip_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
}

int main(int arg, char** argv)
{
  test();
  return 0;
}
