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

// <cuda/std/chrono>

// ceil

// template <class ToDuration, class Rep, class Period>
//   constexpr
//   ToDuration
//   ceil(const duration<Rep, Period>& d);

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

template <class ToDuration, class FromDuration>
__host__ __device__
void
test(const FromDuration& f, const ToDuration& d)
{
    {
    typedef decltype(hip::std::chrono::ceil<ToDuration>(f)) R;
    static_assert((hip::std::is_same<R, ToDuration>::value), "");
    assert(hip::std::chrono::ceil<ToDuration>(f) == d);
    }
}

int main(int, char**)
{
//  7290000ms is 2 hours, 1 minute, and 30 seconds
    test(hip::std::chrono::milliseconds( 7290000), hip::std::chrono::hours( 3));
    test(hip::std::chrono::milliseconds(-7290000), hip::std::chrono::hours(-2));
    test(hip::std::chrono::milliseconds( 7290000), hip::std::chrono::minutes( 122));
    test(hip::std::chrono::milliseconds(-7290000), hip::std::chrono::minutes(-121));

    {
//  9000000ms is 2 hours and 30 minutes
    constexpr hip::std::chrono::hours h1 = hip::std::chrono::ceil<hip::std::chrono::hours>(hip::std::chrono::milliseconds(9000000));
    static_assert(h1.count() == 3, "");
    constexpr hip::std::chrono::hours h2 = hip::std::chrono::ceil<hip::std::chrono::hours>(hip::std::chrono::milliseconds(-9000000));
    static_assert(h2.count() == -2, "");
    }

  return 0;
}
