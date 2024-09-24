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

// test ratio_not_equal

#include <hip/std/ratio>

#include "test_macros.h"

template <class Rat1, class Rat2, bool result>
__host__ __device__
void test()
{
    static_assert((result == hip::std::ratio_not_equal<Rat1, Rat2>::value), "");
#if TEST_STD_VER > 14
    static_assert((result == hip::std::ratio_not_equal_v<Rat1, Rat2>), "");
#endif
}

int main(int, char**)
{
    {
    typedef hip::std::ratio<1, 1> R1;
    typedef hip::std::ratio<1, 1> R2;
    test<R1, R2, false>();
    }
    {
    typedef hip::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef hip::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, false>();
    }
    {
    typedef hip::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef hip::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, false>();
    }
    {
    typedef hip::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef hip::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R2;
    test<R1, R2, false>();
    }
    {
    typedef hip::std::ratio<1, 1> R1;
    typedef hip::std::ratio<1, -1> R2;
    test<R1, R2, true>();
    }
    {
    typedef hip::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef hip::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, true>();
    }
    {
    typedef hip::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef hip::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, true>();
    }
    {
    typedef hip::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef hip::std::ratio<1, -0x7FFFFFFFFFFFFFFFLL> R2;
    test<R1, R2, true>();
    }

  return 0;
}
