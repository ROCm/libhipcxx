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

// UNSUPPORTED: c++98, c++03

// <cuda/std/iterator>

// insert_iterator

// requires CopyConstructible<Cont::value_type>
//   insert_iterator<Cont>&
//   operator=(const Cont::value_type& value);

#include <hip/std/iterator>

#include <hip/std/utility>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#include <hip/std/vector>
#include <hip/std/memory>
#include <hip/std/cassert>

#include "test_macros.h"

template <class C>
__host__ __device__
void
test(C c1, typename C::difference_type j,
     typename C::value_type x1, typename C::value_type x2,
     typename C::value_type x3, const C& c2)
{
    hip::std::insert_iterator<C> q(c1, c1.begin() + j);
    q = hip::std::move(x1);
    q = hip::std::move(x2);
    q = hip::std::move(x3);
    assert(c1 == c2);
}

template <class C>
__host__ __device__
void
insert3at(C& c, typename C::iterator i,
     typename C::value_type x1, typename C::value_type x2,
     typename C::value_type x3)
{
    i = c.insert(i, hip::std::move(x1));
    i = c.insert(++i, hip::std::move(x2));
    c.insert(++i, hip::std::move(x3));
}

struct do_nothing
{
__host__ __device__
    void operator()(void*) const {}
};

int main(int, char**)
{
    {
    typedef hip::std::unique_ptr<int, do_nothing> Ptr;
    typedef hip::std::vector<Ptr> C;
    C c1;
    int x[6] = {0};
    for (int i = 0; i < 3; ++i)
        c1.push_back(Ptr(x+i));
    C c2;
    for (int i = 0; i < 3; ++i)
        c2.push_back(Ptr(x+i));
    insert3at(c2, c2.begin(), Ptr(x+3), Ptr(x+4), Ptr(x+5));
    test(hip::std::move(c1), 0, Ptr(x+3), Ptr(x+4), Ptr(x+5), c2);
    c1.clear();
    for (int i = 0; i < 3; ++i)
        c1.push_back(Ptr(x+i));
    c2.clear();
    for (int i = 0; i < 3; ++i)
        c2.push_back(Ptr(x+i));
    insert3at(c2, c2.begin()+1, Ptr(x+3), Ptr(x+4), Ptr(x+5));
    test(hip::std::move(c1), 1, Ptr(x+3), Ptr(x+4), Ptr(x+5), c2);
    c1.clear();
    for (int i = 0; i < 3; ++i)
        c1.push_back(Ptr(x+i));
    c2.clear();
    for (int i = 0; i < 3; ++i)
        c2.push_back(Ptr(x+i));
    insert3at(c2, c2.begin()+2, Ptr(x+3), Ptr(x+4), Ptr(x+5));
    test(hip::std::move(c1), 2, Ptr(x+3), Ptr(x+4), Ptr(x+5), c2);
    c1.clear();
    for (int i = 0; i < 3; ++i)
        c1.push_back(Ptr(x+i));
    c2.clear();
    for (int i = 0; i < 3; ++i)
        c2.push_back(Ptr(x+i));
    insert3at(c2, c2.begin()+3, Ptr(x+3), Ptr(x+4), Ptr(x+5));
    test(hip::std::move(c1), 3, Ptr(x+3), Ptr(x+4), Ptr(x+5), c2);
    }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif