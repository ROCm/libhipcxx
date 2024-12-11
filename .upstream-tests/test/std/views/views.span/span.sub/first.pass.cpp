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

// template<size_t Count>
//  constexpr span<element_type, Count> first() const;
//
// constexpr span<element_type, dynamic_extent> first(size_type count) const;
//
//  Requires: Count <= size().


#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

template <typename Span, size_t Count>
__host__ __device__
constexpr bool testConstexprSpan(Span sp)
{
    ASSERT_NOEXCEPT(sp.template first<Count>());
    ASSERT_NOEXCEPT(sp.first(Count));
    auto s1 = sp.template first<Count>();
    auto s2 = sp.first(Count);
    using S1 = decltype(s1);
    using S2 = decltype(s2);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
    static_assert(S1::extent == Count, "");
    static_assert(S2::extent == hip::std::dynamic_extent, "");
    return
        s1.data() == s2.data()
     && s1.size() == s2.size();
}


template <typename Span, size_t Count>
__host__ __device__
void testRuntimeSpan(Span sp)
{
    ASSERT_NOEXCEPT(sp.template first<Count>());
    ASSERT_NOEXCEPT(sp.first(Count));
    auto s1 = sp.template first<Count>();
    auto s2 = sp.first(Count);
    using S1 = decltype(s1);
    using S2 = decltype(s2);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
    static_assert(S1::extent == Count, "");
    static_assert(S2::extent == hip::std::dynamic_extent, "");
    assert(s1.data() == s2.data());
    assert(s1.size() == s2.size());
}

__device__ constexpr int carr1[] = {1,2,3,4};
__device__           int   arr[] = {5,6,7};

int main(int, char**)
{
    {
    using Sp = hip::std::span<const int>;
    static_assert(testConstexprSpan<Sp, 0>(Sp{}), "");

    static_assert(testConstexprSpan<Sp, 0>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4>(Sp{carr1}), "");
    }

    {
    using Sp = hip::std::span<const int, 4>;

    static_assert(testConstexprSpan<Sp, 0>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4>(Sp{carr1}), "");
    }
    
    {
    using Sp = hip::std::span<int>;
    testRuntimeSpan<Sp, 0>(Sp{});

    testRuntimeSpan<Sp, 0>(Sp{arr});
    testRuntimeSpan<Sp, 1>(Sp{arr});
    testRuntimeSpan<Sp, 2>(Sp{arr});
    testRuntimeSpan<Sp, 3>(Sp{arr});
    }

    {
    using Sp = hip::std::span<int, 3>;

    testRuntimeSpan<Sp, 0>(Sp{arr});
    testRuntimeSpan<Sp, 1>(Sp{arr});
    testRuntimeSpan<Sp, 2>(Sp{arr});
    testRuntimeSpan<Sp, 3>(Sp{arr});
    }

  return 0;
}
