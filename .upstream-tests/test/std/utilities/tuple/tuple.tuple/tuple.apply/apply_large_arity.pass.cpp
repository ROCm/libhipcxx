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

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: nvrtc
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

// <cuda/std/tuple>

// template <class F, class T> constexpr decltype(auto) apply(F &&, T &&)

// Stress testing large arities with tuple and array.

#include <hip/std/tuple>
#include <hip/std/array>
#include <hip/std/utility>
#include <hip/std/cassert>

#include "test_macros.h"

////////////////////////////////////////////////////////////////////////////////
template <class T, hip::std::size_t Dummy = 0>
struct always_imp
{
    typedef T type;
};

template <class T, hip::std::size_t Dummy = 0>
using always_t = typename always_imp<T, Dummy>::type;

////////////////////////////////////////////////////////////////////////////////
template <class Tuple, class Idx>
struct make_function;

template <class Tp, hip::std::size_t ...Idx>
struct make_function<Tp, hip::std::integer_sequence<hip::std::size_t, Idx...>>
{
    using type = bool (*)(always_t<Tp, Idx>...);
};

template <class Tp, hip::std::size_t Size>
using make_function_t = typename make_function<Tp, hip::std::make_index_sequence<Size>>::type;

////////////////////////////////////////////////////////////////////////////////
template <class Tp, class Idx>
struct make_tuple_imp;

////////////////////////////////////////////////////////////////////////////////
template <class Tp, hip::std::size_t ...Idx>
struct make_tuple_imp<Tp, hip::std::integer_sequence<hip::std::size_t, Idx...>>
{
    using type = hip::std::tuple<always_t<Tp, Idx>...>;
};

template <class Tp, hip::std::size_t Size>
using make_tuple_t = typename make_tuple_imp<Tp, hip::std::make_index_sequence<Size>>::type;

template <class ...Types>
__host__ __device__ bool test_apply_fn(Types...) { return true; }


template <hip::std::size_t Size>
__host__ __device__ void test_all()
{

    using A = hip::std::array<int, Size>;
    using ConstA = hip::std::array<int const, Size>;

    using Tuple = make_tuple_t<int, Size>;
    using CTuple = make_tuple_t<const int, Size>;

    using ValFn  = make_function_t<int, Size>;
    ValFn val_fn = &test_apply_fn;

    using RefFn  = make_function_t<int &, Size>;
    RefFn ref_fn = &test_apply_fn;

    using CRefFn = make_function_t<int const &, Size>;
    CRefFn cref_fn = &test_apply_fn;

    using RRefFn = make_function_t<int &&, Size>;
    RRefFn rref_fn = &test_apply_fn;

    {
        A a{};
        assert(hip::std::apply(val_fn, a));
        assert(hip::std::apply(ref_fn, a));
        assert(hip::std::apply(cref_fn, a));
        assert(hip::std::apply(rref_fn, hip::std::move(a)));
    }
    {
        ConstA a{};
        assert(hip::std::apply(val_fn, a));
        assert(hip::std::apply(cref_fn, a));
    }
    {
        Tuple a{};
        assert(hip::std::apply(val_fn, a));
        assert(hip::std::apply(ref_fn, a));
        assert(hip::std::apply(cref_fn, a));
        assert(hip::std::apply(rref_fn, hip::std::move(a)));
    }
    {
        CTuple a{};
        assert(hip::std::apply(val_fn, a));
        assert(hip::std::apply(cref_fn, a));
    }

}


template <hip::std::size_t Size>
__host__ __device__ void test_one()
{
    using A = hip::std::array<int, Size>;
    using Tuple = make_tuple_t<int, Size>;

    using ValFn  = make_function_t<int, Size>;
    ValFn val_fn = &test_apply_fn;

    {
        A a{};
        assert(hip::std::apply(val_fn, a));
    }
    {
        Tuple a{};
        assert(hip::std::apply(val_fn, a));
    }
}

int main(int, char**)
{
    // Instantiate with 1-5 arguments.
    test_all<1>();
    test_all<2>();
    test_all<3>();
    test_all<4>();
    test_all<5>();

    // Stress test with 256
    test_one<256>();

  return 0;
}
