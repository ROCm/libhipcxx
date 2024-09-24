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

// <cuda/std/chrono>

// duration

// template <class Rep1, class Period1, class Rep2, class Period2>
//   constexpr
//   bool
//   operator==(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

// template <class Rep1, class Period1, class Rep2, class Period2>
//   constexpr
//   bool
//   operator!=(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    hip::std::chrono::seconds s1(3);
    hip::std::chrono::seconds s2(3);
    assert(s1 == s2);
    assert(!(s1 != s2));
    }
    {
    hip::std::chrono::seconds s1(3);
    hip::std::chrono::seconds s2(4);
    assert(!(s1 == s2));
    assert(s1 != s2);
    }
    {
    hip::std::chrono::milliseconds s1(3);
    hip::std::chrono::microseconds s2(3000);
    assert(s1 == s2);
    assert(!(s1 != s2));
    }
    {
    hip::std::chrono::milliseconds s1(3);
    hip::std::chrono::microseconds s2(4000);
    assert(!(s1 == s2));
    assert(s1 != s2);
    }
    {
    hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(9);
    hip::std::chrono::duration<int, hip::std::ratio<3, 5> > s2(10);
    assert(s1 == s2);
    assert(!(s1 != s2));
    }
    {
    hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(10);
    hip::std::chrono::duration<int, hip::std::ratio<3, 5> > s2(9);
    assert(!(s1 == s2));
    assert(s1 != s2);
    }
    {
    hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(9);
    hip::std::chrono::duration<double, hip::std::ratio<3, 5> > s2(10);
    assert(s1 == s2);
    assert(!(s1 != s2));
    }
#if TEST_STD_VER >= 11
    {
    constexpr hip::std::chrono::seconds s1(3);
    constexpr hip::std::chrono::seconds s2(3);
    static_assert(s1 == s2, "");
    static_assert(!(s1 != s2), "");
    }
    {
    constexpr hip::std::chrono::seconds s1(3);
    constexpr hip::std::chrono::seconds s2(4);
    static_assert(!(s1 == s2), "");
    static_assert(s1 != s2, "");
    }
    {
    constexpr hip::std::chrono::milliseconds s1(3);
    constexpr hip::std::chrono::microseconds s2(3000);
    static_assert(s1 == s2, "");
    static_assert(!(s1 != s2), "");
    }
    {
    constexpr hip::std::chrono::milliseconds s1(3);
    constexpr hip::std::chrono::microseconds s2(4000);
    static_assert(!(s1 == s2), "");
    static_assert(s1 != s2, "");
    }
    {
    constexpr hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(9);
    constexpr hip::std::chrono::duration<int, hip::std::ratio<3, 5> > s2(10);
    static_assert(s1 == s2, "");
    static_assert(!(s1 != s2), "");
    }
    {
    constexpr hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(10);
    constexpr hip::std::chrono::duration<int, hip::std::ratio<3, 5> > s2(9);
    static_assert(!(s1 == s2), "");
    static_assert(s1 != s2, "");
    }
    {
    constexpr hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(9);
    constexpr hip::std::chrono::duration<double, hip::std::ratio<3, 5> > s2(10);
    static_assert(s1 == s2, "");
    static_assert(!(s1 != s2), "");
    }
#endif

  return 0;
}
