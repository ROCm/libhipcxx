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

// template <class... UTypes>
//   tuple& operator=(tuple<UTypes...>&& u);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/utility>
#include <hip/std/cassert>

#include "test_macros.h"

struct B
{
    int id_;

    __host__ __device__ explicit B(int i= 0) : id_(i) {}

    __host__ __device__ virtual ~B() {}
};

struct D
    : B
{
    __host__ __device__ explicit D(int i) : B(i) {}
};

struct E {
    E() = default;
    __host__ __device__ E& operator=(int) {
        return *this;
    }
};

int main(int, char**)
{
    {
        typedef hip::std::tuple<long> T0;
        typedef hip::std::tuple<long long> T1;
        T0 t0(2);
        T1 t1;
        t1 = hip::std::move(t0);
        assert(hip::std::get<0>(t1) == 2);
    }
    {
        typedef hip::std::tuple<long, char> T0;
        typedef hip::std::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1;
        t1 = hip::std::move(t0);
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == int('a'));
    }
    {
        typedef hip::std::tuple<long, char, D> T0;
        typedef hip::std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1;
        t1 = hip::std::move(t0);
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == int('a'));
        assert(hip::std::get<2>(t1).id_ == 3);
    }
    {
        D d(3);
        D d2(2);
        typedef hip::std::tuple<long, char, D&> T0;
        typedef hip::std::tuple<long long, int, B&> T1;
        T0 t0(2, 'a', d2);
        T1 t1(1, 'b', d);
        t1 = hip::std::move(t0);
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == int('a'));
        assert(hip::std::get<2>(t1).id_ == 2);
    }
    // unique_ptr unsupported
    /*
    {
        typedef hip::std::tuple<long, char, hip::std::unique_ptr<D>> T0;
        typedef hip::std::tuple<long long, int, hip::std::unique_ptr<B>> T1;
        T0 t0(2, 'a', hip::std::unique_ptr<D>(new D(3)));
        T1 t1;
        t1 = hip::std::move(t0);
        assert(hip::std::get<0>(t1) == 2);
        assert(hip::std::get<1>(t1) == int('a'));
        assert(hip::std::get<2>(t1)->id_ == 3);
    }*/
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
        // Test that tuple evaluates correctly applies an lvalue reference
        // before evaluating is_assignable (ie 'is_assignable<int&, int&&>')
        // instead of evaluating 'is_assignable<int&&, int&&>' which is false.
        int x = 42;
        int y = 43;
        hip::std::tuple<int&&, E> t(hip::std::move(x), E{});
        hip::std::tuple<int&&, int> t2(hip::std::move(y), 44);
        t = hip::std::move(t2);
        assert(hip::std::get<0>(t) == 43);
        assert(&hip::std::get<0>(t) == &x);
    }
#endif
  return 0;
}
