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

// template<class T>    complex<T>           proj(const complex<T>&);
//                      complex<long double> proj(long double);
//                      complex<double>      proj(double);
// template<Integral T> complex<double>      proj(T);
//                      complex<float>       proj(float);

#if defined(_MSC_VER)
#pragma warning(disable: 4244) // conversion from 'const double' to 'int', possible loss of data
#endif

#include <hip/std/complex>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(T x, typename hip::std::enable_if<hip::std::is_integral<T>::value>::type* = 0)
{
    static_assert((hip::std::is_same<decltype(hip::std::proj(x)), hip::std::complex<double> >::value), "");
    assert(hip::std::proj(x) == proj(hip::std::complex<double>(x, 0)));
}

template <class T>
__host__ __device__ void
test(T x, typename hip::std::enable_if<hip::std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((hip::std::is_same<decltype(hip::std::proj(x)), hip::std::complex<T> >::value), "");
    assert(hip::std::proj(x) == proj(hip::std::complex<T>(x, 0)));
}

template <class T>
__host__ __device__ void
test(T x, typename hip::std::enable_if<!hip::std::is_integral<T>::value &&
                                  !hip::std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((hip::std::is_same<decltype(hip::std::proj(x)), hip::std::complex<T> >::value), "");
    assert(hip::std::proj(x) == proj(hip::std::complex<T>(x, 0)));
}

template <class T>
__host__ __device__ void
test()
{
    test<T>(0);
    test<T>(1);
    test<T>(10);
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
    test<int>();
    test<unsigned>();
    test<long long>();

  return 0;
}
