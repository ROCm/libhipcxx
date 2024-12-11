//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// <cuda/std/complex>

// constexpr complex(const T& re = T(), const T& im = T());

#include <hip/std/complex>
#include <hip/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ void
test()
{
    {
    const hip::std::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    }
    {
    const hip::std::complex<T> c = 7.5;
    assert(c.real() == 7.5);
    assert(c.imag() == 0);
    }
    {
    const hip::std::complex<T> c(8.5);
    assert(c.real() == 8.5);
    assert(c.imag() == 0);
    }
    {
    const hip::std::complex<T> c(10.5, -9.5);
    assert(c.real() == 10.5);
    assert(c.imag() == -9.5);
    }
#if TEST_STD_VER >= 11
    {
    constexpr hip::std::complex<T> c;
    static_assert(c.real() == 0, "");
    static_assert(c.imag() == 0, "");
    }
    {
    constexpr hip::std::complex<T> c = 7.5;
    static_assert(c.real() == 7.5, "");
    static_assert(c.imag() == 0, "");
    }
    {
    constexpr hip::std::complex<T> c(8.5);
    static_assert(c.real() == 8.5, "");
    static_assert(c.imag() == 0, "");
    }
    {
    constexpr hip::std::complex<T> c(10.5, -9.5);
    static_assert(c.real() == 10.5, "");
    static_assert(c.imag() == -9.5, "");
    }
#endif
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();

  return 0;
}
