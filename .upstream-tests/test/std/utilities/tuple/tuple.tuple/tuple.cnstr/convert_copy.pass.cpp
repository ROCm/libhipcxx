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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... UTypes> tuple(const tuple<UTypes...>& u);

// XFAIL: gcc-4.8, gcc-4.9

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"

struct Explicit {
  int value;
  __host__ __device__ explicit Explicit(int x) : value(x) {}
};

struct Implicit {
  int value;
  __host__ __device__ Implicit(int x) : value(x) {}
};

struct ExplicitTwo {
    __host__ __device__ ExplicitTwo() {}
    __host__ __device__ ExplicitTwo(ExplicitTwo const&) {}
    __host__ __device__ ExplicitTwo(ExplicitTwo &&) {}

    template <class T, class = typename hip::std::enable_if<!hip::std::is_same<T, ExplicitTwo>::value>::type>
    __host__ __device__ explicit ExplicitTwo(T) {}
};

struct B
{
    int id_;

    __host__ __device__ explicit B(int i) : id_(i) {}
};

struct D
    : B
{
    __host__ __device__ explicit D(int i) : B(i) {}
};

#if TEST_STD_VER > 11

struct A
{
    int id_;

    __host__ __device__ constexpr A(int i) : id_(i) {}
    __host__ __device__ friend constexpr bool operator==(const A& x, const A& y) {return x.id_ == y.id_;}
};

struct C
{
    int id_;

    __host__ __device__ constexpr explicit C(int i) : id_(i) {}
    __host__ __device__ friend constexpr bool operator==(const C& x, const C& y) {return x.id_ == y.id_;}
};

#endif

int main(int, char**)
{
    {
        typedef hip::std::tuple<long> T0;
        typedef hip::std::tuple<long long> T1;
        T0 t0(2);
        T1 t1 = t0;
        assert(hip::std::get<0>(t1) == 2);
    }
#if TEST_STD_VER > 11
    {
        typedef hip::std::tuple<int> T0;
        typedef hip::std::tuple<A> T1;
        constexpr T0 t0(2);
        constexpr T1 t1 = t0;
        static_assert(hip::std::get<0>(t1) == 2, "");
    }
    {
        typedef hip::std::tuple<int> T0;
        typedef hip::std::tuple<C> T1;
        constexpr T0 t0(2);
        constexpr T1 t1{t0};
        static_assert(hip::std::get<0>(t1) == C(2), "");
    }
#endif
    {
        typedef hip::std::tuple<long, char> T0;
        typedef hip::std::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1 = t0;
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == int('a'));
    }
    {
        typedef hip::std::tuple<long, char, D> T0;
        typedef hip::std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1 = t0;
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == int('a'));
        assert(hip::std::get<2>(t1).id_ == 3);
    }
    {
        D d(3);
        typedef hip::std::tuple<long, char, D&> T0;
        typedef hip::std::tuple<long long, int, B&> T1;
        T0 t0(2, 'a', d);
        T1 t1 = t0;
        d.id_ = 2;
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == int('a'));
        assert(hip::std::get<2>(t1).id_ == 2);
    }
    {
        typedef hip::std::tuple<long, char, int> T0;
        typedef hip::std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', 3);
        T1 t1(t0);
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == int('a'));
        assert(hip::std::get<2>(t1).id_ == 3);
    }
    {
        const hip::std::tuple<int> t1(42);
        hip::std::tuple<Explicit> t2(t1);
        assert(hip::std::get<0>(t2).value == 42);
    }
    {
        const hip::std::tuple<int> t1(42);
        hip::std::tuple<Implicit> t2 = t1;
        assert(hip::std::get<0>(t2).value == 42);
    }
    {
        static_assert(hip::std::is_convertible<ExplicitTwo&&, ExplicitTwo>::value, "");
        static_assert(hip::std::is_convertible<hip::std::tuple<ExplicitTwo&&>&&, const hip::std::tuple<ExplicitTwo>&>::value, "");

#if !(defined(_MSC_VER) && _MSC_VER < 1916)
        ExplicitTwo e;
        hip::std::tuple<ExplicitTwo> t = hip::std::tuple<ExplicitTwo&&>(hip::std::move(e));
        ((void)t);
#endif
    }
  return 0;
}
