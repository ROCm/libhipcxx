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

// constexpr reference back() const noexcept;
//   Expects: empty() is false.
//   Effects: Equivalent to: return *(data() + (size() - 1));
//


#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

template <typename Span>
__host__ __device__
constexpr bool testConstexprSpan(Span sp)
{
    ASSERT_NOEXCEPT(sp.back());
    return &sp.back() == sp.data() + (sp.size() - 1);
}

template <typename Span>
__host__ __device__
void testRuntimeSpan(Span sp)
{
    ASSERT_NOEXCEPT(sp.back());
    assert(&sp.back() == sp.data() + (sp.size() - 1));
}

template <typename Span>
__host__ __device__
void testEmptySpan(Span sp)
{
    if (!sp.empty())
        unused(sp.back());
}

struct A{};
__device__ constexpr int iArr1[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};
__device__           int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{
    STATIC_ASSERT_CXX14(testConstexprSpan(hip::std::span<const int>(iArr1, 1)));
    STATIC_ASSERT_CXX14(testConstexprSpan(hip::std::span<const int>(iArr1, 2)));
    STATIC_ASSERT_CXX14(testConstexprSpan(hip::std::span<const int>(iArr1, 3)));
    STATIC_ASSERT_CXX14(testConstexprSpan(hip::std::span<const int>(iArr1, 4)));

    STATIC_ASSERT_CXX14(testConstexprSpan(hip::std::span<const int, 1>(iArr1, 1)));
    STATIC_ASSERT_CXX14(testConstexprSpan(hip::std::span<const int, 2>(iArr1, 2)));
    STATIC_ASSERT_CXX14(testConstexprSpan(hip::std::span<const int, 3>(iArr1, 3)));
    STATIC_ASSERT_CXX14(testConstexprSpan(hip::std::span<const int, 4>(iArr1, 4)));

    testRuntimeSpan(hip::std::span<int>(iArr2, 1));
    testRuntimeSpan(hip::std::span<int>(iArr2, 2));
    testRuntimeSpan(hip::std::span<int>(iArr2, 3));
    testRuntimeSpan(hip::std::span<int>(iArr2, 4));


    testRuntimeSpan(hip::std::span<int, 1>(iArr2, 1));
    testRuntimeSpan(hip::std::span<int, 2>(iArr2, 2));
    testRuntimeSpan(hip::std::span<int, 3>(iArr2, 3));
    testRuntimeSpan(hip::std::span<int, 4>(iArr2, 4));

    hip::std::span<int, 0> sp;
    testEmptySpan(sp);

    return 0;
}
