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

// <complex>

// test cases

#ifndef CASES_H
#define CASES_H

#include <hip/std/complex>
#include <hip/std/cassert>

using testcases_t = hip::std::complex<double>[152];

struct _testcases {
    testcases_t _cases;

    static constexpr size_t count = sizeof(testcases_t) / sizeof(hip::std::complex<double>);

    __host__ __device__  const hip::std::complex<double>* begin() const {
        return  &_cases[0];
    }
    __host__ __device__  const hip::std::complex<double>* cbegin() const {
        return  &_cases[0];
    }
    __host__ __device__  hip::std::complex<double>* begin() {
        return  &_cases[0];
    }

    __host__ __device__  const hip::std::complex<double>* end() const {
        return  &_cases[count];
    }
    __host__ __device__  const hip::std::complex<double>* cend() const {
        return  &_cases[count];
    }
    __host__ __device__  hip::std::complex<double>* end() {
        return  &_cases[count];
    }

    __host__ __device__  hip::std::complex<double>& operator[](size_t n) {
        return _cases[n];
    }

    __host__ __device__  const hip::std::complex<double>& operator[](size_t n) const {
        return _cases[n];
    }
};

__host__ __device__ _testcases get_testcases() {
    _testcases tc {
        hip::std::complex<double>( 1.e-6,  1.e-6),
        hip::std::complex<double>(-1.e-6,  1.e-6),
        hip::std::complex<double>(-1.e-6, -1.e-6),
        hip::std::complex<double>( 1.e-6, -1.e-6),

        hip::std::complex<double>( 1.e+6,  1.e-6),
        hip::std::complex<double>(-1.e+6,  1.e-6),
        hip::std::complex<double>(-1.e+6, -1.e-6),
        hip::std::complex<double>( 1.e+6, -1.e-6),

        hip::std::complex<double>( 1.e-6,  1.e+6),
        hip::std::complex<double>(-1.e-6,  1.e+6),
        hip::std::complex<double>(-1.e-6, -1.e+6),
        hip::std::complex<double>( 1.e-6, -1.e+6),

        hip::std::complex<double>( 1.e+6,  1.e+6),
        hip::std::complex<double>(-1.e+6,  1.e+6),
        hip::std::complex<double>(-1.e+6, -1.e+6),
        hip::std::complex<double>( 1.e+6, -1.e+6),

        hip::std::complex<double>(-0, -1.e-6),
        hip::std::complex<double>(-0,  1.e-6),
        hip::std::complex<double>(-0,  1.e+6),
        hip::std::complex<double>(-0, -1.e+6),
        hip::std::complex<double>( 0, -1.e-6),
        hip::std::complex<double>( 0,  1.e-6),
        hip::std::complex<double>( 0,  1.e+6),
        hip::std::complex<double>( 0, -1.e+6),

        hip::std::complex<double>(-1.e-6, -0),
        hip::std::complex<double>( 1.e-6, -0),
        hip::std::complex<double>( 1.e+6, -0),
        hip::std::complex<double>(-1.e+6, -0),
        hip::std::complex<double>(-1.e-6,  0),
        hip::std::complex<double>( 1.e-6,  0),
        hip::std::complex<double>( 1.e+6,  0),

        hip::std::complex<double>(NAN, NAN),
        hip::std::complex<double>(-INFINITY, NAN),
        hip::std::complex<double>(-2, NAN),
        hip::std::complex<double>(-1, NAN),
        hip::std::complex<double>(-0.5, NAN),
        hip::std::complex<double>(-0., NAN),
        hip::std::complex<double>(+0., NAN),
        hip::std::complex<double>(0.5, NAN),
        hip::std::complex<double>(1, NAN),
        hip::std::complex<double>(2, NAN),
        hip::std::complex<double>(INFINITY, NAN),

        hip::std::complex<double>(NAN, -INFINITY),
        hip::std::complex<double>(-INFINITY, -INFINITY),
        hip::std::complex<double>(-2, -INFINITY),
        hip::std::complex<double>(-1, -INFINITY),
        hip::std::complex<double>(-0.5, -INFINITY),
        hip::std::complex<double>(-0., -INFINITY),
        hip::std::complex<double>(+0., -INFINITY),
        hip::std::complex<double>(0.5, -INFINITY),
        hip::std::complex<double>(1, -INFINITY),
        hip::std::complex<double>(2, -INFINITY),
        hip::std::complex<double>(INFINITY, -INFINITY),

        hip::std::complex<double>(NAN, -2),
        hip::std::complex<double>(-INFINITY, -2),
        hip::std::complex<double>(-2, -2),
        hip::std::complex<double>(-1, -2),
        hip::std::complex<double>(-0.5, -2),
        hip::std::complex<double>(-0., -2),
        hip::std::complex<double>(+0., -2),
        hip::std::complex<double>(0.5, -2),
        hip::std::complex<double>(1, -2),
        hip::std::complex<double>(2, -2),
        hip::std::complex<double>(INFINITY, -2),

        hip::std::complex<double>(NAN, -1),
        hip::std::complex<double>(-INFINITY, -1),
        hip::std::complex<double>(-2, -1),
        hip::std::complex<double>(-1, -1),
        hip::std::complex<double>(-0.5, -1),
        hip::std::complex<double>(-0., -1),
        hip::std::complex<double>(+0., -1),
        hip::std::complex<double>(0.5, -1),
        hip::std::complex<double>(1, -1),
        hip::std::complex<double>(2, -1),
        hip::std::complex<double>(INFINITY, -1),

        hip::std::complex<double>(NAN, -0.5),
        hip::std::complex<double>(-INFINITY, -0.5),
        hip::std::complex<double>(-2, -0.5),
        hip::std::complex<double>(-1, -0.5),
        hip::std::complex<double>(-0.5, -0.5),
        hip::std::complex<double>(-0., -0.5),
        hip::std::complex<double>(+0., -0.5),
        hip::std::complex<double>(0.5, -0.5),
        hip::std::complex<double>(1, -0.5),
        hip::std::complex<double>(2, -0.5),
        hip::std::complex<double>(INFINITY, -0.5),

        hip::std::complex<double>(NAN, -0.),
        hip::std::complex<double>(-INFINITY, -0.),
        hip::std::complex<double>(-2, -0.),
        hip::std::complex<double>(-1, -0.),
        hip::std::complex<double>(-0.5, -0.),
        hip::std::complex<double>(-0., -0.),
        hip::std::complex<double>(+0., -0.),
        hip::std::complex<double>(0.5, -0.),
        hip::std::complex<double>(1, -0.),
        hip::std::complex<double>(2, -0.),
        hip::std::complex<double>(INFINITY, -0.),

        hip::std::complex<double>(NAN, +0.),
        hip::std::complex<double>(-INFINITY, +0.),
        hip::std::complex<double>(-2, +0.),
        hip::std::complex<double>(-1, +0.),
        hip::std::complex<double>(-0.5, +0.),
        hip::std::complex<double>(-0., +0.),
        hip::std::complex<double>(+0., +0.),
        hip::std::complex<double>(0.5, +0.),
        hip::std::complex<double>(1, +0.),
        hip::std::complex<double>(2, +0.),
        hip::std::complex<double>(INFINITY, +0.),

        hip::std::complex<double>(NAN, 0.5),
        hip::std::complex<double>(-INFINITY, 0.5),
        hip::std::complex<double>(-2, 0.5),
        hip::std::complex<double>(-1, 0.5),
        hip::std::complex<double>(-0.5, 0.5),
        hip::std::complex<double>(-0., 0.5),
        hip::std::complex<double>(+0., 0.5),
        hip::std::complex<double>(0.5, 0.5),
        hip::std::complex<double>(1, 0.5),
        hip::std::complex<double>(2, 0.5),
        hip::std::complex<double>(INFINITY, 0.5),

        hip::std::complex<double>(NAN, 1),
        hip::std::complex<double>(-INFINITY, 1),
        hip::std::complex<double>(-2, 1),
        hip::std::complex<double>(-1, 1),
        hip::std::complex<double>(-0.5, 1),
        hip::std::complex<double>(-0., 1),
        hip::std::complex<double>(+0., 1),
        hip::std::complex<double>(0.5, 1),
        hip::std::complex<double>(1, 1),
        hip::std::complex<double>(2, 1),
        hip::std::complex<double>(INFINITY, 1),

        hip::std::complex<double>(NAN, 2),
        hip::std::complex<double>(-INFINITY, 2),
        hip::std::complex<double>(-2, 2),
        hip::std::complex<double>(-1, 2),
        hip::std::complex<double>(-0.5, 2),
        hip::std::complex<double>(-0., 2),
        hip::std::complex<double>(+0., 2),
        hip::std::complex<double>(0.5, 2),
        hip::std::complex<double>(1, 2),
        hip::std::complex<double>(2, 2),
        hip::std::complex<double>(INFINITY, 2),

        hip::std::complex<double>(NAN, INFINITY),
        hip::std::complex<double>(-INFINITY, INFINITY),
        hip::std::complex<double>(-2, INFINITY),
        hip::std::complex<double>(-1, INFINITY),
        hip::std::complex<double>(-0.5, INFINITY),
        hip::std::complex<double>(-0., INFINITY),
        hip::std::complex<double>(+0., INFINITY),
        hip::std::complex<double>(0.5, INFINITY),
        hip::std::complex<double>(1, INFINITY),
        hip::std::complex<double>(2, INFINITY),
        hip::std::complex<double>(INFINITY, INFINITY)
    };

    return tc;
}

enum {zero, non_zero, inf, NaN, non_zero_nan};

template <class T>
__host__ __device__ int
classify(const hip::std::complex<T>& x)
{
    if (x == hip::std::complex<T>())
        return zero;
    if (hip::std::isinf(x.real()) || hip::std::isinf(x.imag()))
        return inf;
    if (hip::std::isnan(x.real()) && hip::std::isnan(x.imag()))
        return NaN;
    if (hip::std::isnan(x.real()))
    {
        if (x.imag() == T(0))
            return NaN;
        return non_zero_nan;
    }
    if (hip::std::isnan(x.imag()))
    {
        if (x.real() == T(0))
            return NaN;
        return non_zero_nan;
    }
    return non_zero;
}

inline
__host__ __device__ int
classify(double x)
{
    if (x == 0)
        return zero;
    if (hip::std::isinf(x))
        return inf;
    if (hip::std::isnan(x))
        return NaN;
    return non_zero;
}

__host__ __device__ void is_about(float x, float y)
{
    assert(hip::std::abs((x-y)/(x+y)) < 1.e-6);
}

__host__ __device__ void is_about(double x, double y)
{
    assert(hip::std::abs((x-y)/(x+y)) < 1.e-14);
}

// CUDA treats long double as double
/*
__host__ __device__ void is_about(long double x, long double y)
{
    assert(hip::std::abs((x-y)/(x+y)) < 1.e-14);
}
*/
#endif  // CASES_H
