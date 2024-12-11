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

// tuple(tuple&& u);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/utility>
#include <hip/std/cassert>

#include "test_macros.h"
#include "MoveOnly.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

struct ConstructsWithTupleLeaf
{
    __host__ __device__ ConstructsWithTupleLeaf() {}

    __host__ __device__ ConstructsWithTupleLeaf(ConstructsWithTupleLeaf const &) { assert(false); }
    __host__ __device__ ConstructsWithTupleLeaf(ConstructsWithTupleLeaf &&) {}

    template <class T>
    __host__ __device__ ConstructsWithTupleLeaf(T) {
        static_assert(!hip::std::is_same<T, T>::value,
                      "Constructor instantiated for type other than int");
    }
};

// move_only type which triggers the empty base optimization
struct move_only_ebo {
  move_only_ebo() = default;
  move_only_ebo(move_only_ebo&&) = default;
};

// a move_only type which does not trigger the empty base optimization
struct move_only_large final {
  __host__ __device__ move_only_large() : value(42) {}
  move_only_large(move_only_large&&) = default;
  int value;
};

template <class Elem>
__host__ __device__ void test_sfinae() {
    using Tup = hip::std::tuple<Elem>;
    // hip::std::allocator not supported
    // using Alloc = hip::std::allocator<void>;
    // using Tag = hip::std::allocator_arg_t;
    // special members
    {
        static_assert(hip::std::is_default_constructible<Tup>::value, "");
        static_assert(hip::std::is_move_constructible<Tup>::value, "");
        static_assert(!hip::std::is_copy_constructible<Tup>::value, "");
        static_assert(!hip::std::is_constructible<Tup, Tup&>::value, "");
    }
    // args constructors
    {
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
        static_assert(hip::std::is_constructible<Tup, Elem&&>::value, "");
#endif
        static_assert(!hip::std::is_constructible<Tup, Elem const&>::value, "");
        static_assert(!hip::std::is_constructible<Tup, Elem&>::value, "");
    }
    // hip::std::allocator not supported
    /*
    // uses-allocator special member constructors
    {
        static_assert(hip::std::is_constructible<Tup, Tag, Alloc>::value, "");
        static_assert(hip::std::is_constructible<Tup, Tag, Alloc, Tup&&>::value, "");
        static_assert(!hip::std::is_constructible<Tup, Tag, Alloc, Tup const&>::value, "");
        static_assert(!hip::std::is_constructible<Tup, Tag, Alloc, Tup &>::value, "");
    }
    // uses-allocator args constructors
    {
        static_assert(hip::std::is_constructible<Tup, Tag, Alloc, Elem&&>::value, "");
        static_assert(!hip::std::is_constructible<Tup, Tag, Alloc, Elem const&>::value, "");
        static_assert(!hip::std::is_constructible<Tup, Tag, Alloc, Elem &>::value, "");
    }
    */
}

int main(int, char**)
{
    {
        typedef hip::std::tuple<> T;
        T t0;
        T t = hip::std::move(t0);
        unused(t); // Prevent unused warning
    }
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
        typedef hip::std::tuple<MoveOnly> T;
        T t0(MoveOnly(0));
        T t = hip::std::move(t0);
        assert(hip::std::get<0>(t) == 0);
    }
    {
        typedef hip::std::tuple<MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1));
        T t = hip::std::move(t0);
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == 1);
    }
    {
        typedef hip::std::tuple<MoveOnly, MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
        T t = hip::std::move(t0);
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == 1);
        assert(hip::std::get<2>(t) == 2);
    }
    // A bug in tuple caused __tuple_leaf to use its explicit converting constructor
    //  as its move constructor. This tests that ConstructsWithTupleLeaf is not called
    // (w/ __tuple_leaf)
    {
        typedef hip::std::tuple<ConstructsWithTupleLeaf> d_t;
        d_t d((ConstructsWithTupleLeaf()));
        d_t d2(static_cast<d_t &&>(d));
        unused(d2);
    }
#endif
    {
        test_sfinae<move_only_ebo>();
        test_sfinae<move_only_large>();
    }

  return 0;
}
