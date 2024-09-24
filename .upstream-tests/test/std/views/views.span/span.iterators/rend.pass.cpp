//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// UNSUPPORTED: c++03, c++11

// <span>

// constexpr       reverse_iterator  rend() const noexcept;

#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

template <typename Span>
__host__ __device__
constexpr bool testConstexprSpan(Span s)
{
    bool ret = true;
    typename Span::reverse_iterator e = s.rend();
    if (s.empty())
    {
        ret = ret &&  (e == s.rbegin());
    }
    else
    {
        ret = ret &&  (e != s.rbegin());
    }

    ret = ret &&  (static_cast<size_t>(e - s.rbegin()) == s.size());
    return ret;
}

template <class Span>
__host__ __device__
void testRuntimeSpan(Span s)
{
    typename Span::reverse_iterator e = s.rend();
    if (s.empty())
    {
        assert(e == s.rbegin());
    }
    else
    {
        assert(e != s.rbegin());
    }

    assert(static_cast<size_t>(e - s.rbegin()) == s.size());
}


struct A{};
__host__ __device__ bool operator==(A, A) {return true;}

#if TEST_STD_VER > 14
__device__ constexpr int iArr1[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};
#endif
__device__           int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{
#if TEST_STD_VER > 14
    static_assert(testConstexprSpan(hip::std::span<int>()),            "");
    static_assert(testConstexprSpan(hip::std::span<long>()),           "");
    static_assert(testConstexprSpan(hip::std::span<double>()),         "");
    static_assert(testConstexprSpan(hip::std::span<A>()),              "");

    static_assert(testConstexprSpan(hip::std::span<int, 0>()),         "");
    static_assert(testConstexprSpan(hip::std::span<long, 0>()),        "");
    static_assert(testConstexprSpan(hip::std::span<double, 0>()),      "");
    static_assert(testConstexprSpan(hip::std::span<A, 0>()),           "");

    static_assert(testConstexprSpan(hip::std::span<const int>(iArr1, 1)),    "");
    static_assert(testConstexprSpan(hip::std::span<const int>(iArr1, 2)),    "");
    static_assert(testConstexprSpan(hip::std::span<const int>(iArr1, 3)),    "");
    static_assert(testConstexprSpan(hip::std::span<const int>(iArr1, 4)),    "");
    static_assert(testConstexprSpan(hip::std::span<const int>(iArr1, 5)),    "");
#endif // TEST_STD_VER > 14

    testRuntimeSpan(hip::std::span<int>        ());
    testRuntimeSpan(hip::std::span<long>       ());
    testRuntimeSpan(hip::std::span<double>     ());
    testRuntimeSpan(hip::std::span<A>          ());

    testRuntimeSpan(hip::std::span<int, 0>        ());
    testRuntimeSpan(hip::std::span<long, 0>       ());
    testRuntimeSpan(hip::std::span<double, 0>     ());
    testRuntimeSpan(hip::std::span<A, 0>          ());

    testRuntimeSpan(hip::std::span<int>(iArr2, 1));
    testRuntimeSpan(hip::std::span<int>(iArr2, 2));
    testRuntimeSpan(hip::std::span<int>(iArr2, 3));
    testRuntimeSpan(hip::std::span<int>(iArr2, 4));
    testRuntimeSpan(hip::std::span<int>(iArr2, 5));

  return 0;
}
