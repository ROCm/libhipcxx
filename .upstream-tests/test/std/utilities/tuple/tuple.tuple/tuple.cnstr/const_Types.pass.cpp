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

// explicit tuple(const T&...);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"

template <class ...>
struct never {
    enum { value = 0 };
};

struct NoValueCtor
{
    STATIC_MEMBER_VAR(count, int);

    __host__ __device__ NoValueCtor() : id(++count()) {}
    __host__ __device__ NoValueCtor(NoValueCtor const & other) : id(other.id) { ++count(); }

    // The constexpr is required to make is_constructible instantiate this template.
    // The explicit is needed to test-around a similar bug with is_convertible.
    template <class T>
    __host__ __device__ constexpr explicit NoValueCtor(T)
    { static_assert(never<T>::value, "This should not be instantiated"); }

    int id;
};


struct NoValueCtorEmpty
{
    __host__ __device__ NoValueCtorEmpty() {}
    __host__ __device__ NoValueCtorEmpty(NoValueCtorEmpty const &) {}

    template <class T>
    __host__ __device__ constexpr explicit NoValueCtorEmpty(T)
    { static_assert(never<T>::value, "This should not be instantiated"); }
};


struct ImplicitCopy {
  __host__ __device__ explicit ImplicitCopy(int) {}
  __host__ __device__ ImplicitCopy(ImplicitCopy const&) {}
};

// Test that tuple(hip::std::allocator_arg, Alloc, Types const&...) allows implicit
// copy conversions in return value expressions.
__host__ __device__ hip::std::tuple<ImplicitCopy> testImplicitCopy1() {
    ImplicitCopy i(42);
    return {i};
}

__host__ __device__ hip::std::tuple<ImplicitCopy> testImplicitCopy2() {
    const ImplicitCopy i(42);
    return {i};
}

__host__ __device__ hip::std::tuple<ImplicitCopy> testImplicitCopy3() {
    const ImplicitCopy i(42);
    return i;
}

int main(int, char**)
{
    NoValueCtor::count() = 0;
    {
        // check that the literal '0' can implicitly initialize a stored pointer.
        hip::std::tuple<int*> t = 0;
        assert(hip::std::get<0>(t) == nullptr);
    }
    {
        hip::std::tuple<int> t(2);
        assert(hip::std::get<0>(t) == 2);
    }
#if TEST_STD_VER > 11
    {
        constexpr hip::std::tuple<int> t(2);
        static_assert(hip::std::get<0>(t) == 2, "");
    }
    {
        constexpr hip::std::tuple<int> t;
        static_assert(hip::std::get<0>(t) == 0, "");
    }
#endif
    {
        hip::std::tuple<int, char*> t(2, 0);
        assert(hip::std::get<0>(t) == 2);
        assert(hip::std::get<1>(t) == nullptr);
    }
#if TEST_STD_VER > 11
    {
        constexpr hip::std::tuple<int, char*> t(2, nullptr);
        static_assert(hip::std::get<0>(t) == 2, "");
        static_assert(hip::std::get<1>(t) == nullptr, "");
    }
#endif
    {
        hip::std::tuple<int, char*> t(2, nullptr);
        assert(hip::std::get<0>(t) == 2);
        assert(hip::std::get<1>(t) == nullptr);
    }
    // hip::std::string not supported
    /*
    {
        hip::std::tuple<int, char*, hip::std::string> t(2, nullptr, "text");
        assert(hip::std::get<0>(t) == 2);
        assert(hip::std::get<1>(t) == nullptr);
        assert(hip::std::get<2>(t) == "text");
    }
    */
    // __tuple_leaf<T> uses is_constructible<T, U> to disable its explicit converting
    // constructor overload __tuple_leaf(U &&). Evaluating is_constructible can cause a compile error.
    // This overload is evaluated when __tuple_leafs copy or move ctor is called.
    // This checks that is_constructible is not evaluated when U == __tuple_leaf.
    {
        hip::std::tuple<int, NoValueCtor, int, int> t(1, NoValueCtor(), 2, 3);
        assert(hip::std::get<0>(t) == 1);
        assert(hip::std::get<1>(t).id == 1);
        assert(hip::std::get<2>(t) == 2);
        assert(hip::std::get<3>(t) == 3);
    }
    {
        hip::std::tuple<int, NoValueCtorEmpty, int, int> t(1, NoValueCtorEmpty(), 2, 3);
        assert(hip::std::get<0>(t) == 1);
        assert(hip::std::get<2>(t) == 2);
        assert(hip::std::get<3>(t) == 3);
    }
    // extensions
    // hip::std::string not supported
    /*
#ifdef _LIBCUDACXX_VERSION
    {
        hip::std::tuple<int, char*, hip::std::string> t(2);
        assert(hip::std::get<0>(t) == 2);
        assert(hip::std::get<1>(t) == nullptr);
        assert(hip::std::get<2>(t) == "");
    }
    {
        hip::std::tuple<int, char*, hip::std::string> t(2, nullptr);
        assert(hip::std::get<0>(t) == 2);
        assert(hip::std::get<1>(t) == nullptr);
        assert(hip::std::get<2>(t) == "");
    }
    {
        hip::std::tuple<int, char*, hip::std::string, double> t(2, nullptr, "text");
        assert(hip::std::get<0>(t) == 2);
        assert(hip::std::get<1>(t) == nullptr);
        assert(hip::std::get<2>(t) == "text");
        assert(hip::std::get<3>(t) == 0.0);
    }
#endif
    */

  return 0;
}
