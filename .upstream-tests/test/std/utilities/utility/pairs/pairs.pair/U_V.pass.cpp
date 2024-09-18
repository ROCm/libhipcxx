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

// template<class U, class V> pair(U&& x, V&& y);


#include <hip/std/utility>
// cuda/std/memory not supported
// #include <hip/std/memory>
#include <hip/std/cassert>

#include "archetypes.h"
#include "test_convertible.h"

#include "test_macros.h"
using namespace ImplicitTypes; // Get implicitly archetypes

template <class T1, class T1Arg,
          bool CanCopy = true, bool CanConvert = CanCopy>
__host__ __device__ void test_sfinae() {
    using P1 = hip::std::pair<T1, int>;
    using P2 = hip::std::pair<int, T1>;
    using T2 = int const&;
    static_assert(hip::std::is_constructible<P1, T1Arg, T2>::value == CanCopy, "");
    static_assert(test_convertible<P1,   T1Arg, T2>() == CanConvert, "");
    static_assert(hip::std::is_constructible<P2, T2,   T1Arg>::value == CanCopy, "");
    static_assert(test_convertible<P2,   T2,   T1Arg>() == CanConvert, "");
}

struct ExplicitT {
  __host__ __device__ constexpr explicit ExplicitT(int x) : value(x) {}
  int value;
};

struct ImplicitT {
  __host__ __device__ constexpr ImplicitT(int x) : value(x) {}
  int value;
};


int main(int, char**)
{
    // cuda/std/memory not supported
    /*
    {
        typedef hip::std::pair<hip::std::unique_ptr<int>, short*> P;
        P p(hip::std::unique_ptr<int>(new int(3)), nullptr);
        assert(*p.first == 3);
        assert(p.second == nullptr);
    }
    */
    {
        // Test non-const lvalue and rvalue types
        test_sfinae<AllCtors, AllCtors&>();
        test_sfinae<AllCtors, AllCtors&&>();
        test_sfinae<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&, true, false>();
        test_sfinae<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&&, true, false>();
        test_sfinae<CopyOnly, CopyOnly&>();
        test_sfinae<CopyOnly, CopyOnly&&>();
        test_sfinae<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&, true, false>();
        test_sfinae<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&&, true, false>();
        test_sfinae<MoveOnly, MoveOnly&, false>();
        test_sfinae<MoveOnly, MoveOnly&&>();
        test_sfinae<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&, false>();
        test_sfinae<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&&, true, false>();
        test_sfinae<NonCopyable, NonCopyable&, false>();
        test_sfinae<NonCopyable, NonCopyable&&, false>();
        test_sfinae<ExplicitTypes::NonCopyable, ExplicitTypes::NonCopyable&, false>();
        test_sfinae<ExplicitTypes::NonCopyable, ExplicitTypes::NonCopyable&&, false>();
    }
    {
        // Test converting types
        test_sfinae<ConvertingType, int&>();
        test_sfinae<ConvertingType, const int&>();
        test_sfinae<ConvertingType, int&&>();
        test_sfinae<ConvertingType, const int&&>();
        test_sfinae<ExplicitTypes::ConvertingType, int&, true, false>();
        test_sfinae<ExplicitTypes::ConvertingType, const int&, true, false>();
        test_sfinae<ExplicitTypes::ConvertingType, int&&, true, false>();
        test_sfinae<ExplicitTypes::ConvertingType, const int&&, true, false>();
    }
#if TEST_STD_VER > 11
    { // explicit constexpr test
        constexpr hip::std::pair<ExplicitT, ExplicitT> p(42, 43);
        static_assert(p.first.value == 42, "");
        static_assert(p.second.value == 43, "");
    }
    { // implicit constexpr test
        constexpr hip::std::pair<ImplicitT, ImplicitT> p = {42, 43};
        static_assert(p.first.value == 42, "");
        static_assert(p.second.value == 43, "");
    }
#endif

  return 0;
}
