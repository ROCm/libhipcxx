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

// <cuda/std/array>

// reference operator[] (size_type)
// const_reference operator[] (size_type); // constexpr in C++14
// reference at (size_type)
// const_reference at (size_type); // constexpr in C++14

#include <hip/std/array>
#include <hip/std/cassert>

#include "test_macros.h"

// hip::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

#if TEST_STD_VER > 14
__host__ __device__ constexpr bool check_idx( size_t idx, double val )
{
    hip::std::array<double, 3> arr = {1, 2, 3.5};
    return arr.at(idx) == val;
}
#endif

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

int main(int, char**)
{
    {
        typedef double T;
        typedef hip::std::array<T, 3> C;
        C c = {1, 2, 3.5};
        C::reference r1 = c.at(0);
        assert(r1 == 1);
        r1 = 5.5;
        assert(c.front() == 5.5);

        C::reference r2 = c.at(2);
        assert(r2 == 3.5);
        r2 = 7.5;
        assert(c.back() == 7.5);

#ifndef TEST_HAS_NO_EXCEPTIONS
        try
        {
            TEST_IGNORE_NODISCARD  c.at(3);
            assert(false);
        }
        catch (const hip::std::out_of_range &) {}
#endif
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        typedef double T;
        typedef hip::std::array<T, 0> C;
        C c = {};
        C const& cc = c;
        try
        {
            TEST_IGNORE_NODISCARD  c.at(0);
            assert(false);
        }
        catch (const hip::std::out_of_range &) {}
        try
        {
            TEST_IGNORE_NODISCARD  cc.at(0);
            assert(false);
        }
        catch (const hip::std::out_of_range &) {}
        unused(c);
    }
#endif
    {
        typedef double T;
        typedef hip::std::array<T, 3> C;
        const C c = {1, 2, 3.5};
        C::const_reference r1 = c.at(0);
        assert(r1 == 1);

        C::const_reference r2 = c.at(2);
        assert(r2 == 3.5);

#ifndef TEST_HAS_NO_EXCEPTIONS
        try
        {
            TEST_IGNORE_NODISCARD  c.at(3);
            assert(false);
        }
        catch (const hip::std::out_of_range &) {}
#endif
        unused(c);
    }

#if TEST_STD_VER > 11
    {
        typedef double T;
        typedef hip::std::array<T, 3> C;
        constexpr C c = {1, 2, 3.5};

#if (!defined(TEST_COMPILER_GCC) || 600 <= TEST_GCC_VER)
// TODO: Investigate this.
        constexpr T t1 = c.at(0);
        static_assert (t1 == 1, "");

        constexpr T t2 = c.at(2);
        static_assert (t2 == 3.5, "");
#endif
        unused(c);
    }
#endif

#if TEST_STD_VER > 14
    {
        static_assert (check_idx(0, 1), "");
        static_assert (check_idx(1, 2), "");
        static_assert (check_idx(2, 3.5), "");
    }
#endif

  return 0;
}
