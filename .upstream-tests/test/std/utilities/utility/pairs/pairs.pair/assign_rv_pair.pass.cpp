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

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair&& p);

#include <hip/std/utility>
// cuda/std/memory not supported
// #include <hip/std/memory>
#include <hip/std/cassert>

#include "test_macros.h"


struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};
struct CopyAssignable {
  CopyAssignable() = default;
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&) = delete;
};
struct MoveAssignable {
  MoveAssignable() = default;
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&) = default;
};

struct CountAssign {
  STATIC_MEMBER_VAR(copied, int);
  STATIC_MEMBER_VAR(moved, int);
  __host__ __device__ static void reset() { copied() = moved() = 0; }
  CountAssign() = default;
  __host__ __device__ CountAssign& operator=(CountAssign const&) { ++copied(); return *this; }
  __host__ __device__ CountAssign& operator=(CountAssign&&) { ++moved(); return *this; }
};

int main(int, char**)
{
    // cuda/std/memory not supported
    /*
    {
        typedef hip::std::pair<hip::std::unique_ptr<int>, int> P;
        P p1(hip::std::unique_ptr<int>(new int(3)), 4);
        P p2;
        p2 = hip::std::move(p1);
        assert(*p2.first == 3);
        assert(p2.second == 4);
    }
    */
    {
        using P = hip::std::pair<int&, int&&>;
        int x = 42;
        int y = 101;
        int x2 = -1;
        int y2 = 300;
        P p1(x, hip::std::move(y));
        P p2(x2, hip::std::move(y2));
        p1 = hip::std::move(p2);
        assert(p1.first == x2);
        assert(p1.second == y2);
    }
    {
        using P = hip::std::pair<int, NonAssignable>;
        static_assert(!hip::std::is_move_assignable<P>::value, "");
    }
    {
        // The move decays to the copy constructor
        CountAssign::reset();
        using P = hip::std::pair<CountAssign, CopyAssignable>;
        static_assert(hip::std::is_move_assignable<P>::value, "");
        P p;
        P p2;
        p = hip::std::move(p2);
        assert(CountAssign::moved() == 0);
        assert(CountAssign::copied() == 1);
    }
    {
        CountAssign::reset();
        using P = hip::std::pair<CountAssign, MoveAssignable>;
        static_assert(hip::std::is_move_assignable<P>::value, "");
        P p;
        P p2;
        p = hip::std::move(p2);
        assert(CountAssign::moved() == 1);
        assert(CountAssign::copied() == 0);
    }

  return 0;
}
