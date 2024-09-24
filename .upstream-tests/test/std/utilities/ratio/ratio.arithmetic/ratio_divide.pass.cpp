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

// test ratio_divide

#include <hip/std/ratio>

int main(int, char**)
{
    {
    typedef hip::std::ratio<1, 1> R1;
    typedef hip::std::ratio<1, 1> R2;
    typedef hip::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == 1 && R::den == 1, "");
    }
    {
    typedef hip::std::ratio<1, 2> R1;
    typedef hip::std::ratio<1, 1> R2;
    typedef hip::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == 1 && R::den == 2, "");
    }
    {
    typedef hip::std::ratio<-1, 2> R1;
    typedef hip::std::ratio<1, 1> R2;
    typedef hip::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 2, "");
    }
    {
    typedef hip::std::ratio<1, -2> R1;
    typedef hip::std::ratio<1, 1> R2;
    typedef hip::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 2, "");
    }
    {
    typedef hip::std::ratio<1, 2> R1;
    typedef hip::std::ratio<-1, 1> R2;
    typedef hip::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 2, "");
    }
    {
    typedef hip::std::ratio<1, 2> R1;
    typedef hip::std::ratio<1, -1> R2;
    typedef hip::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 2, "");
    }
    {
    typedef hip::std::ratio<56987354, 467584654> R1;
    typedef hip::std::ratio<544668, 22145> R2;
    typedef hip::std::ratio_divide<R1, R2>::type R;
    static_assert(R::num == 630992477165LL && R::den == 127339199162436LL, "");
    }

  return 0;
}
