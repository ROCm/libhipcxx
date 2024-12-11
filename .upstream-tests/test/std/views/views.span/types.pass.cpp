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

// template<class ElementType, size_t Extent = dynamic_extent>
// class span {
// public:
//  // constants and types
//  using element_type           = ElementType;
//  using value_type             = remove_cv_t<ElementType>;
//  using size_type              = size_t;
//  using difference_type        = ptrdiff_t;
//  using pointer                = element_type *;
//  using reference              = element_type &;
//  using const_pointe           = const element_type *;
//  using const_reference        = const element_type &;
//  using iterator               = implementation-defined;
//  using reverse_iterator       = std::reverse_iterator<iterator>;
//
//  static constexpr size_type extent = Extent;
//

#include <hip/std/span>
#include <hip/std/cassert>
#include <hip/std/iterator>

#include "test_macros.h"

template <typename S, typename Iter>
__host__ __device__
void testIterator()
{
    typedef hip::std::iterator_traits<Iter> ItT;

    ASSERT_SAME_TYPE(typename ItT::iterator_category, hip::std::random_access_iterator_tag);
    ASSERT_SAME_TYPE(typename ItT::value_type,        typename S::value_type);
    ASSERT_SAME_TYPE(typename ItT::reference,         typename S::reference);
    ASSERT_SAME_TYPE(typename ItT::pointer,           typename S::pointer);
    ASSERT_SAME_TYPE(typename ItT::difference_type,   typename S::difference_type);
}

template <typename S, typename ElementType, hip::std::size_t Size>
__host__ __device__
void testSpan()
{
    ASSERT_SAME_TYPE(typename S::element_type,    ElementType);
    ASSERT_SAME_TYPE(typename S::value_type,      typename hip::std::remove_cv<ElementType>::type);
    ASSERT_SAME_TYPE(typename S::size_type,       hip::std::size_t);
    ASSERT_SAME_TYPE(typename S::difference_type, hip::std::ptrdiff_t);
    ASSERT_SAME_TYPE(typename S::pointer,         ElementType *);
    ASSERT_SAME_TYPE(typename S::const_pointer,   const ElementType *);
    ASSERT_SAME_TYPE(typename S::reference,       ElementType &);
    ASSERT_SAME_TYPE(typename S::const_reference, const ElementType &);

    static_assert(S::extent == Size, ""); // check that it exists

    testIterator<S, typename S::iterator>();
    testIterator<S, typename S::reverse_iterator>();
}


template <typename T>
__host__ __device__
void test()
{
    testSpan<hip::std::span<               T>,                T, hip::std::dynamic_extent>();
    testSpan<hip::std::span<const          T>, const          T, hip::std::dynamic_extent>();
    testSpan<hip::std::span<      volatile T>,       volatile T, hip::std::dynamic_extent>();
    testSpan<hip::std::span<const volatile T>, const volatile T, hip::std::dynamic_extent>();

    testSpan<hip::std::span<               T, 5>,                T, 5>();
    testSpan<hip::std::span<const          T, 5>, const          T, 5>();
    testSpan<hip::std::span<      volatile T, 5>,       volatile T, 5>();
    testSpan<hip::std::span<const volatile T, 5>, const volatile T, 5>();
}

struct A{};

int main(int, char**)
{
    test<int>();
    test<long>();
    test<double>();
    test<A>();

    return 0;
}
