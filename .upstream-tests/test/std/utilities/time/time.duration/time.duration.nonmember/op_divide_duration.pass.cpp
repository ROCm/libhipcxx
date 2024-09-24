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
//   typename common_type<Rep1, Rep2>::type
//   operator/(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"
#include "truncate_fp.h"

int main(int, char**)
{
    {
    hip::std::chrono::nanoseconds ns1(15);
    hip::std::chrono::nanoseconds ns2(5);
    assert(ns1 / ns2 == 3);
    }
    {
    hip::std::chrono::microseconds us1(15);
    hip::std::chrono::nanoseconds ns2(5);
    assert(us1 / ns2 == 3000);
    }
    {
    hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(30);
    hip::std::chrono::duration<int, hip::std::ratio<3, 5> > s2(5);
    assert(s1 / s2 == 6);
    }
    {
    hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(30);
    hip::std::chrono::duration<double, hip::std::ratio<3, 5> > s2(5);
    assert(s1 / s2 == truncate_fp(20./3));
    }
#if TEST_STD_VER >= 11
    {
    constexpr hip::std::chrono::nanoseconds ns1(15);
    constexpr hip::std::chrono::nanoseconds ns2(5);
    static_assert(ns1 / ns2 == 3, "");
    }
    {
    constexpr hip::std::chrono::microseconds us1(15);
    constexpr hip::std::chrono::nanoseconds ns2(5);
    static_assert(us1 / ns2 == 3000, "");
    }
    {
    constexpr hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(30);
    constexpr hip::std::chrono::duration<int, hip::std::ratio<3, 5> > s2(5);
    static_assert(s1 / s2 == 6, "");
    }
    {
    constexpr hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s1(30);
    constexpr hip::std::chrono::duration<double, hip::std::ratio<3, 5> > s2(5);
    static_assert(s1 / s2 == 20./3, "");
    }
#endif

  return 0;
}
