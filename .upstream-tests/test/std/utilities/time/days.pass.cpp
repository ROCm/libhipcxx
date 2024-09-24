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
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>

// using days = duration<signed integer type of at least 25 bits, ratio_multiply<ratio<24>, hours::period>>;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/limits>

#include "test_macros.h"

int main(int, char**)
{
    typedef hip::std::chrono::days D;
    typedef D::rep Rep;
    typedef D::period Period;
    static_assert(hip::std::is_signed<Rep>::value, "");
    static_assert(hip::std::is_integral<Rep>::value, "");
    static_assert(hip::std::numeric_limits<Rep>::digits >= 25, "");
    static_assert(hip::std::is_same_v<Period, hip::std::ratio_multiply<hip::std::ratio<24>, hip::std::chrono::hours::period>>, "");

  return 0;
}
