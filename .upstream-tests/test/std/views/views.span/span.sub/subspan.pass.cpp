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

// template<size_t Offset, size_t Count = dynamic_extent>
//   constexpr span<element_type, see below> subspan() const;
//
// constexpr span<element_type, dynamic_extent> subspan(
//   size_type offset, size_type count = dynamic_extent) const;
//
//  Requires: (0 <= Offset && Offset <= size())
//      && (Count == dynamic_extent || Count >= 0 && Offset + Count <= size())

#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

template <typename Span, size_t Offset, size_t Count>
__host__ __device__
constexpr bool testConstexprSpan(Span sp)
{
    ASSERT_NOEXCEPT(sp.template subspan<Offset, Count>());
    ASSERT_NOEXCEPT(sp.subspan(Offset, Count));
    auto s1 = sp.template subspan<Offset, Count>();
    auto s2 = sp.subspan(Offset, Count);
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

template <typename Span, size_t Offset>
__host__ __device__
constexpr bool testConstexprSpan(Span sp)
{
    ASSERT_NOEXCEPT(sp.template subspan<Offset>());
    ASSERT_NOEXCEPT(sp.subspan(Offset));
    auto s1 = sp.template subspan<Offset>();
    auto s2 = sp.subspan(Offset);
    using S1 = decltype(s1);
    using S2 = decltype(s2);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
    static_assert(S1::extent == (Span::extent == hip::std::dynamic_extent ? hip::std::dynamic_extent : Span::extent - Offset), "");
    static_assert(S2::extent == hip::std::dynamic_extent, "");
    return
        s1.data() == s2.data()
     && s1.size() == s2.size();
}


template <typename Span, size_t Offset, size_t Count>
__host__ __device__
void testRuntimeSpan(Span sp)
{
    ASSERT_NOEXCEPT(sp.template subspan<Offset, Count>());
    ASSERT_NOEXCEPT(sp.subspan(Offset, Count));
    auto s1 = sp.template subspan<Offset, Count>();
    auto s2 = sp.subspan(Offset, Count);
    using S1 = decltype(s1);
    using S2 = decltype(s2);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
    static_assert(S1::extent == Count, "");
    static_assert(S2::extent == hip::std::dynamic_extent, "");
    assert(s1.data() == s2.data());
    assert(s1.size() == s2.size());
}


template <typename Span, size_t Offset>
__host__ __device__
void testRuntimeSpan(Span sp)
{
    ASSERT_NOEXCEPT(sp.template subspan<Offset>());
    ASSERT_NOEXCEPT(sp.subspan(Offset));
    auto s1 = sp.template subspan<Offset>();
    auto s2 = sp.subspan(Offset);
    using S1 = decltype(s1);
    using S2 = decltype(s2);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
    static_assert(S1::extent == (Span::extent == hip::std::dynamic_extent ? hip::std::dynamic_extent : Span::extent - Offset), "");
    static_assert(S2::extent == hip::std::dynamic_extent, "");
    assert(s1.data() == s2.data());
    assert(s1.size() == s2.size());
}

__device__ constexpr int carr1[] = {1,2,3,4};
__device__           int  arr1[] = {5,6,7};

int main(int, char**)
{
    {
    using Sp = hip::std::span<const int>;
    static_assert(testConstexprSpan<Sp, 0>(Sp{}), "");

    static_assert(testConstexprSpan<Sp, 0, 4>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 0>(Sp{carr1}), "");

    static_assert(testConstexprSpan<Sp, 1, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4, 0>(Sp{carr1}), "");
    }

    {
    using Sp = hip::std::span<const int, 4>;

    static_assert(testConstexprSpan<Sp, 0, 4>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 0>(Sp{carr1}), "");

    static_assert(testConstexprSpan<Sp, 1, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4, 0>(Sp{carr1}), "");
    }

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

    testRuntimeSpan<Sp, 0, 3>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 0>(Sp{arr1});

    testRuntimeSpan<Sp, 1, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 2, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 3, 0>(Sp{arr1});
    }

    {
    using Sp = hip::std::span<int, 3>;

    testRuntimeSpan<Sp, 0, 3>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 0>(Sp{arr1});

    testRuntimeSpan<Sp, 1, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 2, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 3, 0>(Sp{arr1});
    }

    {
    using Sp = hip::std::span<int>;
    testRuntimeSpan<Sp, 0>(Sp{});

    testRuntimeSpan<Sp, 0>(Sp{arr1});
    testRuntimeSpan<Sp, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 3>(Sp{arr1});
    }

    {
    using Sp = hip::std::span<int, 3>;

    testRuntimeSpan<Sp, 0>(Sp{arr1});
    testRuntimeSpan<Sp, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 3>(Sp{arr1});
    }

  return 0;
}
