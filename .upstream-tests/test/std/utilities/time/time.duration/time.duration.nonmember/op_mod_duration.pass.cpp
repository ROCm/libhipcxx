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
//   typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2>>::type
//   operator%(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    hip::std::chrono::nanoseconds ns1(15);
    hip::std::chrono::nanoseconds ns2(6);
    hip::std::chrono::nanoseconds r = ns1 % ns2;
    assert(r.count() == 3);
    }
    {
    hip::std::chrono::microseconds us1(15);
    hip::std::chrono::nanoseconds ns2(28);
    hip::std::chrono::nanoseconds r = us1 % ns2;
    assert(r.count() == 20);
    }
    {
    hip::std::chrono::duration<int, hip::std::ratio<3, 5> > s1(6);
    hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s2(3);
    hip::std::chrono::duration<int, hip::std::ratio<1, 15> > r = s1 % s2;
    assert(r.count() == 24);
    }
#if TEST_STD_VER >= 11
    {
    constexpr hip::std::chrono::nanoseconds ns1(15);
    constexpr hip::std::chrono::nanoseconds ns2(6);
    constexpr hip::std::chrono::nanoseconds r = ns1 % ns2;
    static_assert(r.count() == 3, "");
    }
    {
    constexpr hip::std::chrono::microseconds us1(15);
    constexpr hip::std::chrono::nanoseconds ns2(28);
    constexpr hip::std::chrono::nanoseconds r = us1 % ns2;
    static_assert(r.count() == 20, "");
    }
    {
    constexpr hip::std::chrono::duration<int, hip::std::ratio<3, 5> > s1(6);
    constexpr hip::std::chrono::duration<int, hip::std::ratio<2, 3> > s2(3);
    constexpr hip::std::chrono::duration<int, hip::std::ratio<1, 15> > r = s1 % s2;
    static_assert(r.count() == 24, "");
    }
#endif

  return 0;
}
