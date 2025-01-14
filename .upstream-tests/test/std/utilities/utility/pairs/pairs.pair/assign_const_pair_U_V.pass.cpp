//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// UNSUPPORTED: msvc
// UNSUPPORTED: nvrtc

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair& operator=(const pair<U, V>& p);

#include <hip/std/utility>
#include <hip/std/cassert>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "archetypes.h"
#endif

int main(int, char**)
{
    {
        typedef hip::std::pair<int, short> P1;
        typedef hip::std::pair<double, long> P2;
        P1 p1(3, static_cast<short>(4));
        P2 p2;
        p2 = p1;
        assert(p2.first == 3);
        assert(p2.second == 4);
    }
#if TEST_STD_VER >= 11
    {
       using C = TestTypes::TestType;
       using P = hip::std::pair<int, C>;
       using T = hip::std::pair<long, C>;
       const T t(42, -42);
       P p(101, 101);
       C::reset_constructors();
       p = t;
       assert(C::constructed() == 0);
       assert(C::assigned() == 1);
       assert(C::copy_assigned() == 1);
       assert(C::move_assigned() == 0);
       assert(p.first == 42);
       assert(p.second.value == -42);
    }
#endif

  return 0;
}
