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

// move_iterator

// move_iterator operator++(int);
//
//  constexpr in C++17

#include <hip/std/iterator>
#include <hip/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
__host__ __device__
void
test(It i, It x)
{
    hip::std::move_iterator<It> r(i);
    hip::std::move_iterator<It> rr = r++;
    assert(r.base() == x);
    assert(rr.base() == i);
}

int main(int, char**)
{
    char s[] = "123";
    test(input_iterator<char*>(s), input_iterator<char*>(s+1));
    test(forward_iterator<char*>(s), forward_iterator<char*>(s+1));
    test(bidirectional_iterator<char*>(s), bidirectional_iterator<char*>(s+1));
    test(random_access_iterator<char*>(s), random_access_iterator<char*>(s+1));
    test(s, s+1);

#if TEST_STD_VER > 14
    {
    constexpr const char *p = "123456789";
    typedef hip::std::move_iterator<const char *> MI;
    constexpr MI it1 = hip::std::make_move_iterator(p);
    constexpr MI it2 = hip::std::make_move_iterator(p+1);
    static_assert(it1 != it2, "");
    constexpr MI it3 = hip::std::make_move_iterator(p) ++;
    static_assert(it1 == it3, "");
    static_assert(it2 != it3, "");
    }
#endif

  return 0;
}