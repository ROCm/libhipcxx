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

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const T& x, const complex<U>& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const U& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const complex<U>& y);

#if defined(_MSC_VER)
#pragma warning(disable: 4244) // conversion from 'const double' to 'int', possible loss of data
#endif

#include <hip/std/complex>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ double
promote(T, typename hip::std::enable_if<hip::std::is_integral<T>::value>::type* = 0);

__host__ __device__ float promote(float);
__host__ __device__ double promote(double);
__host__ __device__ long double promote(long double);

template <class T, class U>
__host__ __device__ void
test(T x, const hip::std::complex<U>& y)
{
    typedef decltype(promote(x)+promote(real(y))) V;
    static_assert((hip::std::is_same<decltype(hip::std::pow(x, y)), hip::std::complex<V> >::value), "");
    assert(hip::std::pow(x, y) == pow(hip::std::complex<V>(x, 0), hip::std::complex<V>(y)));
}

template <class T, class U>
__host__ __device__ void
test(const hip::std::complex<T>& x, U y)
{
    typedef decltype(promote(real(x))+promote(y)) V;
    static_assert((hip::std::is_same<decltype(hip::std::pow(x, y)), hip::std::complex<V> >::value), "");
    assert(hip::std::pow(x, y) == pow(hip::std::complex<V>(x), hip::std::complex<V>(y, 0)));
}

template <class T, class U>
__host__ __device__ void
test(const hip::std::complex<T>& x, const hip::std::complex<U>& y)
{
    typedef decltype(promote(real(x))+promote(real(y))) V;
    static_assert((hip::std::is_same<decltype(hip::std::pow(x, y)), hip::std::complex<V> >::value), "");
    assert(hip::std::pow(x, y) == pow(hip::std::complex<V>(x), hip::std::complex<V>(y)));
}

template <class T, class U>
__host__ __device__ void
test(typename hip::std::enable_if<hip::std::is_integral<T>::value>::type* = 0, typename hip::std::enable_if<!hip::std::is_integral<U>::value>::type* = 0)
{
    test(T(3), hip::std::complex<U>(4, 5));
    test(hip::std::complex<U>(3, 4), T(5));
}

template <class T, class U>
__host__ __device__ void
test(typename hip::std::enable_if<!hip::std::is_integral<T>::value>::type* = 0, typename hip::std::enable_if<!hip::std::is_integral<U>::value>::type* = 0)
{
    test(T(3), hip::std::complex<U>(4, 5));
    test(hip::std::complex<T>(3, 4), U(5));
    test(hip::std::complex<T>(3, 4), hip::std::complex<U>(5, 6));
}

int main(int, char**)
{
    test<int, float>();
    test<int, double>();

    test<unsigned, float>();
    test<unsigned, double>();

    test<long long, float>();
    test<long long, double>();

    test<float, double>();

    test<double, float>();

// CUDA treats long double as double
//  test<int, long double>();
//  test<unsigned, long double>();
//  test<long long, long double>();
//  test<float, long double>();
//  test<double, long double>();
//  test<long double, float>();
//  test<long double, double>();

  return 0;
}
