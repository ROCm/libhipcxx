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
//   log(const complex<T>& x);

#include <hip/std/complex>
#include <hip/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const hip::std::complex<T>& c, hip::std::complex<T> x)
{
    assert(log(c) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(hip::std::complex<T>(0, 0), hip::std::complex<T>(-INFINITY, 0));
}

__host__ __device__ void test_edges()
{
    const double pi = hip::std::atan2(+0., -0.);
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        hip::std::complex<double> r = log(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            if (hip::std::signbit(testcases[i].real()))
            {
                assert(hip::std::isinf(r.real()));
                assert(r.real() < 0);
                if (hip::std::signbit(testcases[i].imag()))
                    is_about(r.imag(), -pi);
                else
                    is_about(r.imag(), pi);
            }
            else
            {
                assert(hip::std::isinf(r.real()));
                assert(r.real() < 0);
                assert(r.imag() == 0);
                assert(hip::std::signbit(testcases[i].imag()) == hip::std::signbit(r.imag()));
            }
        }
        else if (hip::std::isfinite(testcases[i].real()) && hip::std::isinf(testcases[i].imag()))
        {
            assert(hip::std::isinf(r.real()));
            assert(r.real() > 0);
            if (testcases[i].imag() > 0)
                is_about(r.imag(), pi/2);
            else
                is_about(r.imag(), -pi/2);
        }
        else if (hip::std::isfinite(testcases[i].real()) && hip::std::isnan(testcases[i].imag()))
        {
            assert(hip::std::isnan(r.real()));
            assert(hip::std::isnan(r.imag()));
        }
        else if (hip::std::isinf(testcases[i].real()) && testcases[i].real() < 0 && hip::std::isfinite(testcases[i].imag()))
        {
            assert(hip::std::isinf(r.real()) && r.real() > 0);
            if (r.imag() > 0)
                is_about(r.imag(), pi);
            else
                is_about(r.imag(), -pi);
        }
        else if (hip::std::isinf(testcases[i].real()) && testcases[i].real() > 0 && hip::std::isfinite(testcases[i].imag()))
        {
            assert(hip::std::isinf(r.real()) && r.real() > 0);
            assert(r.imag() == 0);
            assert(hip::std::signbit(testcases[i].imag()) == hip::std::signbit(r.imag()));
        }
        else if (testcases[i].real() == 1 && testcases[i].imag() == 0)
        {
            assert(r.real() == 0);
            assert(hip::std::signbit(r.imag()) == hip::std::signbit(testcases[i].imag()));
        }
        else if (testcases[i].real() == 0 && testcases[i].imag() == 1)
        {
            assert(r.real() == 0);
            is_about(r.imag(), pi/2);
        }
        else if (testcases[i].real() == -1 && testcases[i].imag() == 0)
        {
            assert(r.real() == 0);
            if (hip::std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi);
            else
                is_about(r.imag(),  pi);
        }
        else if (testcases[i].real() == 0 && testcases[i].imag() == -1)
        {
            assert(r.real() == 0);
            is_about(r.imag(), -pi/2);
        }
        else if (hip::std::isfinite(testcases[i].real()) && hip::std::isfinite(testcases[i].imag()) && abs(testcases[i]) < 1)
        {
            assert( hip::std::signbit(r.real()));
            assert(hip::std::signbit(r.imag()) == hip::std::signbit(testcases[i].imag()));
        }
        else if (hip::std::isfinite(testcases[i].real()) && hip::std::isfinite(testcases[i].imag()) && abs(testcases[i]) > 1)
        {
            assert(!hip::std::signbit(r.real()));
            assert(hip::std::signbit(r.imag()) == hip::std::signbit(testcases[i].imag()));
        }
    }
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
    test_edges();

  return 0;
}
