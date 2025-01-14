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
//   operator-(const complex<T>& lhs, const complex<T>& rhs);

#include <hip/std/complex>
#include <hip/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ void
test(const hip::std::complex<T>& lhs, const hip::std::complex<T>& rhs, hip::std::complex<T> x)
{
    assert(lhs - rhs == x);
}

template <class T>
__host__ __device__ void
test()
{
    {
    hip::std::complex<T> lhs(1.5, 2.5);
    hip::std::complex<T> rhs(3.5, 4.5);
    hip::std::complex<T>   x(-2.0, -2.0);
    test(lhs, rhs, x);
    }
    {
    hip::std::complex<T> lhs(1.5, -2.5);
    hip::std::complex<T> rhs(-3.5, 4.5);
    hip::std::complex<T>   x(5.0, -7.0);
    test(lhs, rhs, x);
    }
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();

  return 0;
}
