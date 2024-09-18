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

// test ratio:  The static data members num and den shall have the common
//    divisor of the absolute values of N and D:

#include <hip/std/ratio>

template <long long N, long long D, long long eN, long long eD>
__host__ __device__
void test()
{
    static_assert((hip::std::ratio<N, D>::num == eN), "");
    static_assert((hip::std::ratio<N, D>::den == eD), "");
}

int main(int, char**)
{
    test<1, 1, 1, 1>();
    test<1, 10, 1, 10>();
    test<10, 10, 1, 1>();
    test<10, 1, 10, 1>();
    test<12, 4, 3, 1>();
    test<12, -4, -3, 1>();
    test<-12, 4, -3, 1>();
    test<-12, -4, 3, 1>();
    test<4, 12, 1, 3>();
    test<4, -12, -1, 3>();
    test<-4, 12, -1, 3>();
    test<-4, -12, 1, 3>();
    test<222, 333, 2, 3>();
    test<222, -333, -2, 3>();
    test<-222, 333, -2, 3>();
    test<-222, -333, 2, 3>();
    test<0x7FFFFFFFFFFFFFFFLL, 127, 72624976668147841LL, 1>();
    test<-0x7FFFFFFFFFFFFFFFLL, 127, -72624976668147841LL, 1>();
    test<0x7FFFFFFFFFFFFFFFLL, -127, -72624976668147841LL, 1>();
    test<-0x7FFFFFFFFFFFFFFFLL, -127, 72624976668147841LL, 1>();

  return 0;
}
