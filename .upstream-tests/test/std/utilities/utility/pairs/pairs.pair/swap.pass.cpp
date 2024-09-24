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

// <utility>

// template <class T1, class T2> struct pair

// void swap(pair& p);

#include <hip/std/utility>
#include <hip/std/cassert>

#include "test_macros.h"

struct S {
    int i;
    __host__ __device__ S() : i(0) {}
    __host__ __device__ S(int j) : i(j) {}
    __host__ __device__ S * operator& () { assert(false); return this; }
    __host__ __device__ S const * operator& () const { assert(false); return this; }
    __host__ __device__ bool operator==(int x) const { return i == x; }
};

int main(int, char**)
{
    {
        typedef hip::std::pair<int, short> P1;
        P1 p1(3, static_cast<short>(4));
        P1 p2(5, static_cast<short>(6));
        p1.swap(p2);
        assert(p1.first == 5);
        assert(p1.second == 6);
        assert(p2.first == 3);
        assert(p2.second == 4);
    }
    {
        typedef hip::std::pair<int, S> P1;
        P1 p1(3, S(4));
        P1 p2(5, S(6));
        p1.swap(p2);
        assert(p1.first == 5);
        assert(p1.second == 6);
        assert(p2.first == 3);
        assert(p2.second == 4);
    }

  return 0;
}
