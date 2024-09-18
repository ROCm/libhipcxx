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
//   explicit tuple(UTypes&&... u);

// XFAIL: gcc-4.8, gcc-4.9
// XFAIL: msvc-19.12, msvc-19.13
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

// UNSUPPORTED: c++98, c++03

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"
#include "test_convertible.h"
#include "MoveOnly.h"

#if TEST_STD_VER > 11

struct Empty {};
struct A
{
    int id_;
    __host__ __device__ explicit constexpr A(int i) : id_(i) {}
};

#endif

struct NoDefault { NoDefault() = delete; };

// Make sure the _Up... constructor SFINAEs out when the types that
// are not explicitly initialized are not all default constructible.
// Otherwise, hip::std::is_constructible would return true but instantiating
// the constructor would fail.
__host__ __device__ void test_default_constructible_extension_sfinae()
{
    {
        typedef hip::std::tuple<MoveOnly, NoDefault> Tuple;

        static_assert(!hip::std::is_constructible<
            Tuple,
            MoveOnly
        >::value, "");

        static_assert(hip::std::is_constructible<
            Tuple,
            MoveOnly, NoDefault
        >::value, "");
    }
    {
        typedef hip::std::tuple<MoveOnly, MoveOnly, NoDefault> Tuple;

        static_assert(!hip::std::is_constructible<
            Tuple,
            MoveOnly, MoveOnly
        >::value, "");

        static_assert(hip::std::is_constructible<
            Tuple,
            MoveOnly, MoveOnly, NoDefault
        >::value, "");
    }
    {
        // Same idea as above but with a nested tuple type.
        typedef hip::std::tuple<MoveOnly, NoDefault> Tuple;
        typedef hip::std::tuple<MoveOnly, Tuple, MoveOnly, MoveOnly> NestedTuple;

        static_assert(!hip::std::is_constructible<
            NestedTuple,
            MoveOnly, MoveOnly, MoveOnly, MoveOnly
        >::value, "");

        static_assert(hip::std::is_constructible<
            NestedTuple,
            MoveOnly, Tuple, MoveOnly, MoveOnly
        >::value, "");
    }
    // testing extensions
#ifdef _LIBCUDACXX_VERSION
    {
        typedef hip::std::tuple<MoveOnly, int> Tuple;
        typedef hip::std::tuple<MoveOnly, Tuple, MoveOnly, MoveOnly> NestedTuple;

        static_assert(hip::std::is_constructible<
            NestedTuple,
            MoveOnly, MoveOnly, MoveOnly, MoveOnly
        >::value, "");

        static_assert(hip::std::is_constructible<
            NestedTuple,
            MoveOnly, Tuple, MoveOnly, MoveOnly
        >::value, "");
    }
#endif
}

int main(int, char**)
{
    {
        hip::std::tuple<MoveOnly> t(MoveOnly(0));
        assert(hip::std::get<0>(t) == 0);
    }
    {
        hip::std::tuple<MoveOnly, MoveOnly> t(MoveOnly(0), MoveOnly(1));
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == 1);
    }
    {
        hip::std::tuple<MoveOnly, MoveOnly, MoveOnly> t(MoveOnly(0),
                                                   MoveOnly(1),
                                                   MoveOnly(2));
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == 1);
        assert(hip::std::get<2>(t) == 2);
    }
    // extensions, MSVC issues
#if defined(_LIBCUDACXX_VERSION) && !defined(_MSC_VER)
    {
        using E = MoveOnly;
        using Tup = hip::std::tuple<E, E, E>;
        // Test that the reduced arity initialization extension is only
        // allowed on the explicit constructor.
        static_assert(test_convertible<Tup, E, E, E>(), "");

        Tup t(E(0), E(1));
        static_assert(!test_convertible<Tup, E, E>(), "");
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == 1);
        assert(hip::std::get<2>(t) == MoveOnly());

        Tup t2(E(0));
        static_assert(!test_convertible<Tup, E>(), "");
        assert(hip::std::get<0>(t2) == 0);
        assert(hip::std::get<1>(t2) == E());
        assert(hip::std::get<2>(t2) == E());
    }
#endif
#if TEST_STD_VER > 11
    {
        constexpr hip::std::tuple<Empty> t0{Empty()};
        (void)t0;
    }
    {
        constexpr hip::std::tuple<A, A> t(3, 2);
        static_assert(hip::std::get<0>(t).id_ == 3, "");
    }
#endif
    // Check that SFINAE is properly applied with the default reduced arity
    // constructor extensions.
    test_default_constructible_extension_sfinae();

  return 0;
}
