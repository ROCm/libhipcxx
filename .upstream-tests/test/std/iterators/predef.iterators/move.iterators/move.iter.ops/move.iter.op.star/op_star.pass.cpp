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

// reference operator*() const;
//
//  constexpr in C++17

#include <hip/std/iterator>
#include <hip/std/cassert>
#if defined(_LIBCUDACXX_HAS_MEMORY)
#include <hip/std/memory>
#endif

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
    hip::std::move_iterator<It> r(i);
    assert(*r == x);
    typename hip::std::iterator_traits<It>::value_type x2 = *r;
    assert(x2 == x);
}

struct do_nothing
{
__host__ __device__
    void operator()(void*) const {}
};


int main(int, char**)
{
    {
        A a;
        test(&a, A());
    }
#if defined(_LIBCUDACXX_HAS_MEMORY)
#if TEST_STD_VER >= 11
    {
        int i;
        hip::std::unique_ptr<int, do_nothing> p(&i);
        test(&p, hip::std::unique_ptr<int, do_nothing>(&i));
    }
#endif
#endif
#if TEST_STD_VER > 14
    {
    constexpr const char *p = "123456789";
    typedef hip::std::move_iterator<const char *> MI;
    constexpr MI it1 = hip::std::make_move_iterator(p);
    constexpr MI it2 = hip::std::make_move_iterator(p+1);
    static_assert(*it1 == p[0], "");
    static_assert(*it2 == p[1], "");
    }
#endif

  return 0;
}
