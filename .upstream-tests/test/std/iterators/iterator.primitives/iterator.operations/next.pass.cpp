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

// <cuda/std/iterator>

// template <InputIterator Iter>
//   Iter next(Iter x, Iter::difference_type n = 1);

// LWG #2353 relaxed the requirement on next from ForwardIterator to InputIterator

#include <hip/std/iterator>
#include <hip/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
__host__ __device__
void
test(It i, typename hip::std::iterator_traits<It>::difference_type n, It x)
{
    assert(hip::std::next(i, n) == x);

    It (*next)(It, typename hip::std::iterator_traits<It>::difference_type) = hip::std::next;
    assert(next(i, n) == x);
}

template <class It>
__host__ __device__
void
test(It i, It x)
{
    assert(hip::std::next(i) == x);
}

#if TEST_STD_VER > 14
template <class It>
__host__ __device__
constexpr bool
constexpr_test(It i, typename hip::std::iterator_traits<It>::difference_type n, It x)
{
    return hip::std::next(i, n) == x;
}

template <class It>
__host__ __device__
constexpr bool
constexpr_test(It i, It x)
{
    return hip::std::next(i) == x;
}
#endif

int main(int, char**)
{
    {
    const char* s = "1234567890";
    test(input_iterator<const char*>(s),             10, input_iterator<const char*>(s+10));
    test(forward_iterator<const char*>(s),           10, forward_iterator<const char*>(s+10));
    test(bidirectional_iterator<const char*>(s),     10, bidirectional_iterator<const char*>(s+10));
    test(bidirectional_iterator<const char*>(s+10), -10, bidirectional_iterator<const char*>(s));
    test(random_access_iterator<const char*>(s),     10, random_access_iterator<const char*>(s+10));
    test(random_access_iterator<const char*>(s+10), -10, random_access_iterator<const char*>(s));
    test(s, 10, s+10);

    test(input_iterator<const char*>(s), input_iterator<const char*>(s+1));
    test(forward_iterator<const char*>(s), forward_iterator<const char*>(s+1));
    test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s+1));
    test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s+1));
    test(s, s+1);
    }
#if TEST_STD_VER > 14
    {
    constexpr const char* s = "1234567890";
    static_assert( constexpr_test(input_iterator<const char*>(s),             10, input_iterator<const char*>(s+10)), "" );
    static_assert( constexpr_test(forward_iterator<const char*>(s),           10, forward_iterator<const char*>(s+10)), "" );
    static_assert( constexpr_test(bidirectional_iterator<const char*>(s),     10, bidirectional_iterator<const char*>(s+10)), "" );
    static_assert( constexpr_test(bidirectional_iterator<const char*>(s+10), -10, bidirectional_iterator<const char*>(s)), "" );
    static_assert( constexpr_test(random_access_iterator<const char*>(s),     10, random_access_iterator<const char*>(s+10)), "" );
    static_assert( constexpr_test(random_access_iterator<const char*>(s+10), -10, random_access_iterator<const char*>(s)), "" );
    static_assert( constexpr_test(s, 10, s+10), "" );

    static_assert( constexpr_test(input_iterator<const char*>(s), input_iterator<const char*>(s+1)), "" );
    static_assert( constexpr_test(forward_iterator<const char*>(s), forward_iterator<const char*>(s+1)), "" );
    static_assert( constexpr_test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s+1)), "" );
    static_assert( constexpr_test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s+1)), "" );
    static_assert( constexpr_test(s, s+1), "" );
    }
#endif

  return 0;
}
