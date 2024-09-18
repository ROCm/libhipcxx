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

// constexpr pointer operator->() const;
//
// constexpr in C++17

// Be sure to respect LWG 198:
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#198
// LWG 198 was superseded by LWG 2360
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2360


#include <hip/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#include <hip/std/list>
#endif
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
    int get() const {return data_;}

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
    assert(r->get() == x.get());
}

class B
{
    int data_;
public:
__host__ __device__
    B(int d=1) : data_(d) {}
__host__ __device__
    ~B() {data_ = -1;}

__host__ __device__
    int get() const {return data_;}

__host__ __device__
    friend bool operator==(const B& x, const B& y)
        {return x.data_ == y.data_;}
__host__ __device__
    const B *operator&() const { return nullptr; }
__host__ __device__
    B       *operator&()       { return nullptr; }
};

class C
{
    int data_;
public:
__host__ __device__
    TEST_CONSTEXPR C() : data_(1) {}

__host__ __device__
    TEST_CONSTEXPR int get() const {return data_;}

__host__ __device__
    friend TEST_CONSTEXPR bool operator==(const C& x, const C& y)
        {return x.data_ == y.data_;}
};

TEST_CONSTEXPR  C gC;

int main(int, char**)
{
    A a;
    test(&a+1, A());

#if defined(_LIBCUDACXX_HAS_LIST)
    {
    hip::std::list<B> l;
    l.push_back(B(0));
    l.push_back(B(1));
    l.push_back(B(2));

    {
    hip::std::list<B>::const_iterator i = l.begin();
    assert ( i->get() == 0 );  ++i;
    assert ( i->get() == 1 );  ++i;
    assert ( i->get() == 2 );  ++i;
    assert ( i == l.end ());
    }

    {
    hip::std::list<B>::const_reverse_iterator ri = l.rbegin();
    assert ( ri->get() == 2 );  ++ri;
    assert ( ri->get() == 1 );  ++ri;
    assert ( ri->get() == 0 );  ++ri;
    assert ( ri == l.rend ());
    }
    }
#endif

#if TEST_STD_VER > 14
    {
        #ifndef __CUDA_ARCH__
        typedef hip::std::reverse_iterator<const C *> RI;
        constexpr RI it1 = hip::std::make_reverse_iterator(&gC+1);

        static_assert(it1->get() == gC.get(), "");
        #endif
    }
#endif
    {
        ((void)gC);
    }

  return 0;
}
