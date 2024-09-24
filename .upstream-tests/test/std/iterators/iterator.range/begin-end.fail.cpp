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

#include "test_macros.h"

#if TEST_STD_VER < 11
#error
#else

// <cuda/std/iterator>
// template <class C> auto begin(C& c) -> decltype(c.begin());
// template <class C> auto begin(const C& c) -> decltype(c.begin());
// template <class C> auto end(C& c) -> decltype(c.end());
// template <class C> auto end(const C& c) -> decltype(c.end());
// template <class E> reverse_iterator<const E*> rbegin(initializer_list<E> il);
// template <class E> reverse_iterator<const E*> rend(initializer_list<E> il);


#include <hip/std/iterator>
#include <hip/std/cassert>

namespace Foo {
    struct FakeContainer {};
    typedef int FakeIter;

__host__ __device__
    FakeIter begin(const FakeContainer &)   { return 1; }
__host__ __device__
    FakeIter end  (const FakeContainer &)   { return 2; }
__host__ __device__
    FakeIter rbegin(const FakeContainer &)  { return 3; }
__host__ __device__
    FakeIter rend  (const FakeContainer &)  { return 4; }

__host__ __device__
    FakeIter cbegin(const FakeContainer &)  { return 11; }
__host__ __device__
    FakeIter cend  (const FakeContainer &)  { return 12; }
__host__ __device__
    FakeIter crbegin(const FakeContainer &) { return 13; }
__host__ __device__
    FakeIter crend  (const FakeContainer &) { return 14; }
}


int main(int, char**) {
// Bug #28927 - shouldn't find these via ADL
    TEST_IGNORE_NODISCARD  hip::std::cbegin (Foo::FakeContainer());
    TEST_IGNORE_NODISCARD  hip::std::cend   (Foo::FakeContainer());
    TEST_IGNORE_NODISCARD  hip::std::crbegin(Foo::FakeContainer());
    TEST_IGNORE_NODISCARD  hip::std::crend  (Foo::FakeContainer());

  return 0;
}
#endif
