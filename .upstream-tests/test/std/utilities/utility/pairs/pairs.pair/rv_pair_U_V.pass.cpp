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
// UNSUPPORTED: msvc
// UNSUPPORTED: nvrtc
// XFAIL: gcc-4

// <utility>

// template <class T1, class T2> struct pair

// template <class U, class V> pair(pair<U, V>&& p);

#include <hip/std/utility>
// cuda/std/memory not supported
// #include <hip/std/memory>
#include <hip/std/cassert>

#include "archetypes.h"
#include "test_convertible.h"

#include "test_macros.h"
using namespace ImplicitTypes; // Get implicitly archetypes

template <class T1, class U1,
          bool CanCopy = true, bool CanConvert = CanCopy>
__host__ __device__ void test_pair_rv()
{
    using P1 = hip::std::pair<T1, int>;
    using P2 = hip::std::pair<int, T1>;
    using UP1 = hip::std::pair<U1, int>&&;
    using UP2 = hip::std::pair<int, U1>&&;
    static_assert(hip::std::is_constructible<P1, UP1>::value == CanCopy, "");
    static_assert(test_convertible<P1, UP1>() == CanConvert, "");
    static_assert(hip::std::is_constructible<P2, UP2>::value == CanCopy, "");
    static_assert(test_convertible<P2,  UP2>() == CanConvert, "");
}

struct Base
{
    __host__ __device__ virtual ~Base() {}
};

struct Derived
    : public Base
{
};


template <class T, class U>
struct DPair : public hip::std::pair<T, U> {
  using Base = hip::std::pair<T, U>;
  using Base::Base;
};

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
        typedef hip::std::pair<hip::std::unique_ptr<Derived>, int> P1;
        typedef hip::std::pair<hip::std::unique_ptr<Base>, long> P2;
        P1 p1(hip::std::unique_ptr<Derived>(), 4);
        P2 p2 = hip::std::move(p1);
        assert(p2.first == nullptr);
        assert(p2.second == 4);
    }
    */
    {
        // We allow derived types to use this constructor
        using P1 = DPair<long, long>;
        using P2 = hip::std::pair<int, int>;
        P1 p1(42, 101);
        P2 p2(hip::std::move(p1));
        assert(p2.first == 42);
        assert(p2.second == 101);
    }
    {
        test_pair_rv<AllCtors, AllCtors>();
        test_pair_rv<AllCtors, AllCtors&>();
        test_pair_rv<AllCtors, AllCtors&&>();
        test_pair_rv<AllCtors, const AllCtors&>();
        test_pair_rv<AllCtors, const AllCtors&&>();

        test_pair_rv<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors>();
        test_pair_rv<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&, true, false>();
        test_pair_rv<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&&, true, false>();
        test_pair_rv<ExplicitTypes::AllCtors, const ExplicitTypes::AllCtors&, true, false>();
        test_pair_rv<ExplicitTypes::AllCtors, const ExplicitTypes::AllCtors&&, true, false>();

        test_pair_rv<MoveOnly, MoveOnly>();
        test_pair_rv<MoveOnly, MoveOnly&, false>();
        test_pair_rv<MoveOnly, MoveOnly&&>();

        test_pair_rv<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly>(); // copy construction
        test_pair_rv<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&, false>();
        test_pair_rv<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&&, true, false>();

        test_pair_rv<CopyOnly, CopyOnly>();
        test_pair_rv<CopyOnly, CopyOnly&>();
        test_pair_rv<CopyOnly, CopyOnly&&>();

        test_pair_rv<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly>();
        test_pair_rv<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&, true, false>();
        test_pair_rv<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&&, true, false>();

        test_pair_rv<NonCopyable, NonCopyable, false>();
        test_pair_rv<NonCopyable, NonCopyable&, false>();
        test_pair_rv<NonCopyable, NonCopyable&&, false>();
        test_pair_rv<NonCopyable, const NonCopyable&, false>();
        test_pair_rv<NonCopyable, const NonCopyable&&, false>();
    }
    { // Test construction of references
        test_pair_rv<NonCopyable&, NonCopyable&>();
        test_pair_rv<NonCopyable&, NonCopyable&&>();
        test_pair_rv<NonCopyable&, NonCopyable const&, false>();
        test_pair_rv<NonCopyable const&, NonCopyable&&>();
        test_pair_rv<NonCopyable&&, NonCopyable&&>();

        test_pair_rv<ConvertingType&, int, false>();
        test_pair_rv<ExplicitTypes::ConvertingType&, int, false>();
        // Unfortunately the below conversions are allowed and create dangling
        // references.
        //test_pair_rv<ConvertingType&&, int>();
        //test_pair_rv<ConvertingType const&, int>();
        //test_pair_rv<ConvertingType const&&, int>();
        // But these are not because the converting constructor is explicit.
        test_pair_rv<ExplicitTypes::ConvertingType&&, int, false>();
        test_pair_rv<ExplicitTypes::ConvertingType const&, int, false>();
        test_pair_rv<ExplicitTypes::ConvertingType const&&, int, false>();
    }
    {
        test_pair_rv<AllCtors, int, false>();
        test_pair_rv<ExplicitTypes::AllCtors, int, false>();
        test_pair_rv<ConvertingType, int>();
        test_pair_rv<ExplicitTypes::ConvertingType, int, true, false>();

        test_pair_rv<ConvertingType, int>();
        test_pair_rv<ConvertingType, ConvertingType>();
        test_pair_rv<ConvertingType, ConvertingType const&>();
        test_pair_rv<ConvertingType, ConvertingType&>();
        test_pair_rv<ConvertingType, ConvertingType&&>();

        test_pair_rv<ExplicitTypes::ConvertingType, int, true, false>();
        test_pair_rv<ExplicitTypes::ConvertingType, int&, true, false>();
        test_pair_rv<ExplicitTypes::ConvertingType, const int&, true, false>();
        test_pair_rv<ExplicitTypes::ConvertingType, int&&, true, false>();
        test_pair_rv<ExplicitTypes::ConvertingType, const int&&, true, false>();

        test_pair_rv<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType>();
        test_pair_rv<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType const&, true, false>();
        test_pair_rv<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType&, true, false>();
        test_pair_rv<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType&&, true, false>();
    }
#if TEST_STD_VER > 11
    { // explicit constexpr test
        constexpr hip::std::pair<int, int> p1(42, 43);
        constexpr hip::std::pair<ExplicitT, ExplicitT> p2(hip::std::move(p1));
        static_assert(p2.first.value == 42, "");
        static_assert(p2.second.value == 43, "");
    }
    { // implicit constexpr test
        constexpr hip::std::pair<int, int> p1(42, 43);
        constexpr hip::std::pair<ImplicitT, ImplicitT> p2 = hip::std::move(p1);
        static_assert(p2.first.value == 42, "");
        static_assert(p2.second.value == 43, "");
    }
#endif

  return 0;
}
