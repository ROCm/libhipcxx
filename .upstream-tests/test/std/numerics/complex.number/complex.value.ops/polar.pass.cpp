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
//   polar(const T& rho, const T& theta = T());  // changed from '0' by LWG#2870

#include <hip/std/complex>
#include <hip/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const T& rho, hip::std::complex<T> x)
{
    assert(hip::std::polar(rho) == x);
}

template <class T>
__host__ __device__ void
test(const T& rho, const T& theta, hip::std::complex<T> x)
{
    assert(hip::std::polar(rho, theta) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(T(0), hip::std::complex<T>(0, 0));
    test(T(1), hip::std::complex<T>(1, 0));
    test(T(100), hip::std::complex<T>(100, 0));
    test(T(0), T(0), hip::std::complex<T>(0, 0));
    test(T(1), T(0), hip::std::complex<T>(1, 0));
    test(T(100), T(0), hip::std::complex<T>(100, 0));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        double r = real(testcases[i]);
        double theta = imag(testcases[i]);
        hip::std::complex<double> z = hip::std::polar(r, theta);
        switch (classify(r))
        {
        case zero:
            if (hip::std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
            {
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
            }
            else
            {
                assert(z == hip::std::complex<double>());
            }
            break;
        case non_zero:
            if (hip::std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
            {
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
            }
            else
            {
                is_about(hip::std::abs(z), r);
            }
            break;
        case inf:
            if (r < 0)
            {
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
            }
            else
            {
                assert(classify(z) == inf);
                if (classify(theta) != NaN && classify(theta) != inf)
                {
                    assert(classify(real(z)) != NaN);
                    assert(classify(imag(z)) != NaN);
                }
            }
            break;
        case NaN:
        case non_zero_nan:
            {
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
            }
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
