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

// tuple& operator=(const tuple& u);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};
struct CopyAssignable {
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable &&) = delete;
};
static_assert(hip::std::is_copy_assignable<CopyAssignable>::value, "");
struct MoveAssignable {
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&) = default;
};

int main(int, char**)
{
    {
        typedef hip::std::tuple<> T;
        T t0;
        T t;
        t = t0;
        unused(t);
    }
    {
        typedef hip::std::tuple<int> T;
        T t0(2);
        T t;
        t = t0;
        assert(hip::std::get<0>(t) == 2);
    }
    {
        typedef hip::std::tuple<int, char> T;
        T t0(2, 'a');
        T t;
        t = t0;
        assert(hip::std::get<0>(t) == 2);
        assert(hip::std::get<1>(t) == 'a');
    }
    // hip::std::string not supported
    /*
    {
        typedef hip::std::tuple<int, char, hip::std::string> T;
        const T t0(2, 'a', "some text");
        T t;
        t = t0;
        assert(hip::std::get<0>(t) == 2);
        assert(hip::std::get<1>(t) == 'a');
        assert(hip::std::get<2>(t) == "some text");
    }
    */
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
        // test reference assignment.
        using T = hip::std::tuple<int&, int&&>;
        int x = 42;
        int y = 100;
        int x2 = -1;
        int y2 = 500;
        T t(x, hip::std::move(y));
        T t2(x2, hip::std::move(y2));
        t = t2;
        assert(hip::std::get<0>(t) == x2);
        assert(&hip::std::get<0>(t) == &x);
        assert(hip::std::get<1>(t) == y2);
        assert(&hip::std::get<1>(t) == &y);
    }
#endif
    // hip::std::unique_ptr not supported
    /*
    {
        // test that the implicitly generated copy assignment operator
        // is properly deleted
        using T = hip::std::tuple<hip::std::unique_ptr<int>>;
        static_assert(!hip::std::is_copy_assignable<T>::value, "");
    }
    */
    {
        using T = hip::std::tuple<int, NonAssignable>;
        static_assert(!hip::std::is_copy_assignable<T>::value, "");
    }
    {
        using T = hip::std::tuple<int, CopyAssignable>;
        static_assert(hip::std::is_copy_assignable<T>::value, "");
    }
    {
        using T = hip::std::tuple<int, MoveAssignable>;
        static_assert(!hip::std::is_copy_assignable<T>::value, "");
    }

  return 0;
}
