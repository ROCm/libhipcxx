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
// template<class T>
//   complex<T>
//   proj(const complex<T>& x);

#include <hip/std/complex>
#include <hip/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const hip::std::complex<T>& z, hip::std::complex<T> x)
{
    assert(proj(z) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(hip::std::complex<T>(1, 2), hip::std::complex<T>(1, 2));
    test(hip::std::complex<T>(-1, 2), hip::std::complex<T>(-1, 2));
    test(hip::std::complex<T>(1, -2), hip::std::complex<T>(1, -2));
    test(hip::std::complex<T>(-1, -2), hip::std::complex<T>(-1, -2));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        hip::std::complex<double> r = proj(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
        case non_zero:
            assert(r == testcases[i]);
            assert(hip::std::signbit(real(r)) == hip::std::signbit(real(testcases[i])));
            assert(hip::std::signbit(imag(r)) == hip::std::signbit(imag(testcases[i])));
            break;
        case inf:
            assert(hip::std::isinf(real(r)) && real(r) > 0);
            assert(imag(r) == 0);
            assert(hip::std::signbit(imag(r)) == hip::std::signbit(imag(testcases[i])));
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
// CUDA treats long double as double
//  test<long double>();();
    test_edges();

  return 0;
}
