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

// reverse_iterator

// constexpr reference operator*() const;
//
// constexpr in C++17

// Be sure to respect LWG 198:
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#198
// LWG 198 was superseded by LWG 2360
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2360

#include <hip/std/iterator>
#include <hip/std/cassert>

#include "test_macros.h"

class A
{
    int data_;
public:
__host__ __device__
    A() : data_(1) {}
__host__ __device__
    ~A() {data_ = -1;}

__host__ __device__
    friend bool operator==(const A& x, const A& y)
        {return x.data_ == y.data_;}
};

template <class It>
__host__ __device__
void
test(It i, typename hip::std::iterator_traits<It>::value_type x)
{
    hip::std::reverse_iterator<It> r(i);
    assert(*r == x);
}

int main(int, char**)
{
    A a;
    test(&a+1, A());

#if TEST_STD_VER > 14
    {
        constexpr const char *p = "123456789";
        typedef hip::std::reverse_iterator<const char *> RI;
        constexpr RI it1 = hip::std::make_reverse_iterator(p+1);
        constexpr RI it2 = hip::std::make_reverse_iterator(p+2);
        static_assert(*it1 == p[0], "");
        static_assert(*it2 == p[1], "");
    }
#endif

  return 0;
}
