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

// <utility>

// template <class T1, class T2> struct pair

// Test the SFINAE required by LWG Issue #2367.
// is_default_constructible<pair>

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: msvc

#include <hip/std/utility>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

struct DeletedDefault {
    // A class with a deleted default constructor. Used to test the SFINAE
    // on hip::std::pair's default constructor.
    __host__ __device__ constexpr explicit DeletedDefault(int x) : value(x) {}
    __host__ __device__ constexpr DeletedDefault() = delete;
    int value;
};

template <class Tp, bool>
struct DependantType: public Tp {};

template <class T, bool Val>
using DependantIsDefault = DependantType<hip::std::is_default_constructible<T>, Val>;

template <class T>
struct DefaultSFINAES {
    template <bool Dummy = false, class = typename hip::std::enable_if<
             DependantIsDefault<T, Dummy>::value
                >::type
            >
    __host__ __device__ constexpr DefaultSFINAES() : value() {}
    __host__ __device__ constexpr explicit DefaultSFINAES(T const& x) : value(x) {}
    T value;
};

struct NoDefault {
    __host__ __device__ constexpr NoDefault(int v) : value(v) {}
    int value;
};

template <class Tp>
__host__ __device__ void test_not_is_default_constructible()
{
    {
        typedef hip::std::pair<int, Tp> P;
        static_assert(!hip::std::is_default_constructible<P>::value, "");
        static_assert(hip::std::is_constructible<P, int, Tp>::value, "");
    }
    {
        typedef hip::std::pair<Tp, int> P;
        static_assert(!hip::std::is_default_constructible<P>::value, "");
        static_assert(hip::std::is_constructible<P, Tp, int>::value, "");
    }
    {
        typedef hip::std::pair<Tp, Tp> P;
        static_assert(!hip::std::is_default_constructible<P>::value, "");
        static_assert(hip::std::is_constructible<P, Tp, Tp>::value, "");
    }
}

template <class Tp>
__host__ __device__ void test_is_default_constructible()
{
    {
        typedef hip::std::pair<int, Tp> P;
        static_assert(hip::std::is_default_constructible<P>::value, "");
    }
    {
        typedef hip::std::pair<Tp, int> P;
        static_assert(hip::std::is_default_constructible<P>::value, "");
    }
    {
        typedef hip::std::pair<Tp, Tp> P;
        static_assert(hip::std::is_default_constructible<P>::value, "");
    }
}

template <class T>
struct IllFormedDefaultImp {
  __host__ __device__ constexpr explicit IllFormedDefaultImp(int v) : value(v) {}
  __host__ __device__ constexpr IllFormedDefaultImp() : value(T::DoesNotExistAndShouldNotCompile) {}
  int value;
};

typedef IllFormedDefaultImp<int> IllFormedDefault;
    // A class which provides a constexpr default constructor with a valid
    // signature but an ill-formed body. The A compile error will be emitted if
    // the default constructor is instantiated.


// Check that the SFINAE on the default constructor is not evaluated when
// it isn't needed. If the default constructor of 'IllFormedDefault' is evaluated
// in C++11, even with is_default_constructible, then this test should fail to
// compile. In C++14 and greater evaluate each test is evaluated as a constant
// expression.
// See LWG issue #2367
__host__ __device__ void test_illformed_default()
{
    {
    typedef hip::std::pair<IllFormedDefault, int> P;
    static_assert((hip::std::is_constructible<P, IllFormedDefault, int>::value), "");
    TEST_CONSTEXPR_CXX14 P p(IllFormedDefault(42), -5);
    STATIC_ASSERT_CXX14(p.first.value == 42 && p.second == -5);
    }
    {
    typedef hip::std::pair<int, IllFormedDefault> P;
    static_assert((hip::std::is_constructible<P, int, IllFormedDefault>::value), "");
    TEST_CONSTEXPR_CXX14 IllFormedDefault dd(-5);
    TEST_CONSTEXPR_CXX14 P p(42, dd);
    STATIC_ASSERT_CXX14(p.first == 42 && p.second.value == -5);
    }
    {
    typedef hip::std::pair<IllFormedDefault, IllFormedDefault> P;
    static_assert((hip::std::is_constructible<P, IllFormedDefault, IllFormedDefault>::value), "");
    TEST_CONSTEXPR_CXX14 P p(IllFormedDefault(42), IllFormedDefault(-5));
    STATIC_ASSERT_CXX14(p.first.value == 42 && p.second.value == -5);
    }
}


int main(int, char**)
{
    {
        // Check that pair<T, U> can still be used even if
        // is_default_constructible<T> or is_default_constructible<U> cause
        // a compilation error.
        test_illformed_default();
    }
    {
        // pair::pair() is only disable in C++11 and beyond.
        test_not_is_default_constructible<NoDefault>();
        test_not_is_default_constructible<DeletedDefault>();
        test_not_is_default_constructible<DefaultSFINAES<int&>>();
        test_not_is_default_constructible<DefaultSFINAES<int&&>>();
        test_not_is_default_constructible<int&>();
        test_not_is_default_constructible<int&&>();
    }
    {
        test_is_default_constructible<int>();
        test_is_default_constructible<DefaultSFINAES<int>>();
    }

  return 0;
}
