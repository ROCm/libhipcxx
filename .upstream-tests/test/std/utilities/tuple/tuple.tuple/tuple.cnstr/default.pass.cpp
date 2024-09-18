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

// explicit(see-below) constexpr tuple();

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"
#include "DefaultOnly.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

struct NoDefault {
    NoDefault() = delete;
    __host__ __device__ explicit NoDefault(int) { }
};

struct NoExceptDefault {
    NoExceptDefault() noexcept = default;
};

struct ThrowingDefault {
    __host__ __device__ ThrowingDefault() { }
};

struct IllFormedDefault {
    __host__ __device__ IllFormedDefault(int x) : value(x) {}
    template <bool Pred = false>
    __host__ __device__ constexpr IllFormedDefault() {
        static_assert(Pred,
            "The default constructor should not be instantiated");
    }
    int value;
};

int main(int, char**)
{
    {
        hip::std::tuple<> t;
        unused(t);
    }
    {
        hip::std::tuple<int> t;
        assert(hip::std::get<0>(t) == 0);
    }
    {
        hip::std::tuple<int, char*> t;
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == nullptr);
    }
    // hip::std::string not supported
    /*
    {
        hip::std::tuple<int, char*, hip::std::string> t;
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == nullptr);
        assert(hip::std::get<2>(t) == "");
    }
    {
        hip::std::tuple<int, char*, hip::std::string, DefaultOnly> t;
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == nullptr);
        assert(hip::std::get<2>(t) == "");
        assert(hip::std::get<3>(t) == DefaultOnly());
    }
    */
    {
        // See bug #21157.
        static_assert(!hip::std::is_default_constructible<hip::std::tuple<NoDefault>>(), "");
        static_assert(!hip::std::is_default_constructible<hip::std::tuple<DefaultOnly, NoDefault>>(), "");
        static_assert(!hip::std::is_default_constructible<hip::std::tuple<NoDefault, DefaultOnly, NoDefault>>(), "");
    }
    {
        static_assert(noexcept(hip::std::tuple<NoExceptDefault>()), "");
        static_assert(noexcept(hip::std::tuple<NoExceptDefault, NoExceptDefault>()), "");

        static_assert(!noexcept(hip::std::tuple<ThrowingDefault, NoExceptDefault>()), "");
        static_assert(!noexcept(hip::std::tuple<NoExceptDefault, ThrowingDefault>()), "");
        static_assert(!noexcept(hip::std::tuple<ThrowingDefault, ThrowingDefault>()), "");
    }
    {
        constexpr hip::std::tuple<> t;
        unused(t);
    }
    {
        constexpr hip::std::tuple<int> t;
        assert(hip::std::get<0>(t) == 0);
    }
    {
        constexpr hip::std::tuple<int, char*> t;
        assert(hip::std::get<0>(t) == 0);
        assert(hip::std::get<1>(t) == nullptr);
    }
    {
    // Check that the SFINAE on the default constructor is not evaluated when
    // it isn't needed. If the default constructor is evaluated then this test
    // should fail to compile.
        IllFormedDefault v(0);
        hip::std::tuple<IllFormedDefault> t(v);
        unused(t);
    }
    {
        struct Base { };
        struct Derived : Base { protected: Derived() = default; };
        static_assert(!hip::std::is_default_constructible<hip::std::tuple<Derived, int> >::value, "");
    }

    return 0;
}
