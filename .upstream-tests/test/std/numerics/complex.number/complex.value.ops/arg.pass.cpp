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
//   T
//   arg(const complex<T>& x);

#include <hip/std/complex>
#include <hip/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test()
{
    hip::std::complex<T> z(1, 0);
    assert(arg(z) == 0);
}

__host__ __device__ void test_edges()
{
    const double pi = hip::std::atan2(+0., -0.);
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        double r = arg(testcases[i]);
        if (hip::std::isnan(testcases[i].real()) || hip::std::isnan(testcases[i].imag()))
            assert(hip::std::isnan(r));
        else
        {
            switch (classify(testcases[i]))
            {
            case zero:
                if (hip::std::signbit(testcases[i].real()))
                {
                    if (hip::std::signbit(testcases[i].imag()))
                        is_about(r, -pi);
                    else
                        is_about(r, pi);
                }
                else
                {
                    assert(hip::std::signbit(testcases[i].imag()) == hip::std::signbit(r));
                }
                break;
            case non_zero:
                if (testcases[i].real() == 0)
                {
                    if (testcases[i].imag() < 0)
                        is_about(r, -pi/2);
                    else
                        is_about(r, pi/2);
                }
                else if (testcases[i].imag() == 0)
                {
                    if (testcases[i].real() < 0)
                    {
                        if (hip::std::signbit(testcases[i].imag()))
                            is_about(r, -pi);
                        else
                            is_about(r, pi);
                    }
                    else
                    {
                        assert(r == 0);
                        assert(hip::std::signbit(testcases[i].imag()) == hip::std::signbit(r));
                    }
                }
                else if (testcases[i].imag() > 0)
                    assert(r > 0);
                else
                    assert(r < 0);
                break;
            case inf:
                if (hip::std::isinf(testcases[i].real()) && hip::std::isinf(testcases[i].imag()))
                {
                    if (testcases[i].real() < 0)
                    {
                        if (testcases[i].imag() > 0)
                            is_about(r, 0.75 * pi);
                        else
                            is_about(r, -0.75 * pi);
                    }
                    else
                    {
                        if (testcases[i].imag() > 0)
                            is_about(r, 0.25 * pi);
                        else
                            is_about(r, -0.25 * pi);
                    }
                }
                else if (hip::std::isinf(testcases[i].real()))
                {
                    if (testcases[i].real() < 0)
                    {
                        if (hip::std::signbit(testcases[i].imag()))
                            is_about(r, -pi);
                        else
                            is_about(r, pi);
                    }
                    else
                    {
                        assert(r == 0);
                        assert(hip::std::signbit(r) == hip::std::signbit(testcases[i].imag()));
                    }
                }
                else
                {
                    if (testcases[i].imag() < 0)
                        is_about(r, -pi/2);
                    else
                        is_about(r, pi/2);
                }
                break;
            }
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
