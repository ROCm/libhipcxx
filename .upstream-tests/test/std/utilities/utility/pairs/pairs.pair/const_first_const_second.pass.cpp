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
// UNSUPPORTED: nvrtc
// XFAIL: gcc-4

// <utility>

// template <class T1, class T2> struct pair

// pair(const T1& x, const T2& y);

#include <hip/std/utility>
#include <hip/std/cassert>

#include "archetypes.h"
#include "test_convertible.h"

#include "test_macros.h"
using namespace ImplicitTypes; // Get implicitly archetypes

struct ExplicitT {
  __host__ __device__ constexpr explicit ExplicitT(int x) : value(x) {}
  __host__ __device__ constexpr explicit ExplicitT(ExplicitT const& o) : value(o.value) {}
  int value;
};

struct ImplicitT {
  __host__ __device__ constexpr ImplicitT(int x) : value(x) {}
  __host__ __device__ constexpr ImplicitT(ImplicitT const& o) : value(o.value) {}
  int value;
};

template <class T1,
          bool CanCopy = true, bool CanConvert = CanCopy>
__host__ __device__ void test_sfinae() {
    using P1 = hip::std::pair<T1, int>;
    using P2 = hip::std::pair<int, T1>;
    using T1Arg = T1 const&;
    using T2 = int const&;
    static_assert(hip::std::is_constructible<P1, T1Arg, T2>::value == CanCopy, "");
    static_assert(test_convertible<P1,   T1Arg, T2>() == CanConvert, "");
    static_assert(hip::std::is_constructible<P2, T2,   T1Arg>::value == CanCopy, "");
    static_assert(test_convertible<P2,   T2,   T1Arg>() == CanConvert, "");
}

int main(int, char**)
{
    {
        typedef hip::std::pair<float, short*> P;
        P p(3.5f, 0);
        assert(p.first == 3.5f);
        assert(p.second == nullptr);
    }
    {
        typedef hip::std::pair<ImplicitT, int> P;
        P p(1, 2);
        assert(p.first.value == 1);
        assert(p.second == 2);
    }
    {
        test_sfinae<AllCtors>();
        test_sfinae<ExplicitTypes::AllCtors, true, false>();
        test_sfinae<CopyOnly>();
        test_sfinae<ExplicitTypes::CopyOnly, true, false>();
        test_sfinae<MoveOnly, false>();
        test_sfinae<ExplicitTypes::MoveOnly, false>();
        test_sfinae<NonCopyable, false>();
        test_sfinae<ExplicitTypes::NonCopyable, false>();
    }
#if TEST_STD_VER > 11
    {
        typedef hip::std::pair<float, short*> P;
        constexpr P p(3.5f, 0);
        static_assert(p.first == 3.5f, "");
        static_assert(p.second == nullptr, "");
    }
    {
        using P = hip::std::pair<ExplicitT, int>;
        constexpr ExplicitT e(42);
        constexpr int x = 10;
        constexpr P p(e, x);
        static_assert(p.first.value == 42, "");
        static_assert(p.second == 10, "");
    }
    {
        using P = hip::std::pair<ImplicitT, int>;
        constexpr ImplicitT e(42);
        constexpr int x = 10;
        constexpr P p = {e, x};
        static_assert(p.first.value == 42, "");
        static_assert(p.second == 10, "");
    }
#endif

  return 0;
}
