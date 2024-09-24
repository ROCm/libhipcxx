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

// test numeric_limits

// The default numeric_limits<T> template shall have all members, but with
// 0 or false values.

#include <hip/std/limits>
#include <hip/std/cassert>

#include "test_macros.h"

struct A
{
    __host__ __device__
    A(int i = 0) : data_(i) {}
    int data_;
};

__host__ __device__
bool operator == (const A& x, const A& y) {return x.data_ == y.data_;}

int main(int, char**)
{
    static_assert(hip::std::numeric_limits<A>::is_specialized == false,
                 "hip::std::numeric_limits<A>::is_specialized == false");
    assert(hip::std::numeric_limits<A>::min() == A());
    assert(hip::std::numeric_limits<A>::max() == A());
    assert(hip::std::numeric_limits<A>::lowest() == A());
    static_assert(hip::std::numeric_limits<A>::digits == 0,
                 "hip::std::numeric_limits<A>::digits == 0");
    static_assert(hip::std::numeric_limits<A>::digits10 == 0,
                 "hip::std::numeric_limits<A>::digits10 == 0");
    static_assert(hip::std::numeric_limits<A>::max_digits10 == 0,
                 "hip::std::numeric_limits<A>::max_digits10 == 0");
    static_assert(hip::std::numeric_limits<A>::is_signed == false,
                 "hip::std::numeric_limits<A>::is_signed == false");
    static_assert(hip::std::numeric_limits<A>::is_integer == false,
                 "hip::std::numeric_limits<A>::is_integer == false");
    static_assert(hip::std::numeric_limits<A>::is_exact == false,
                 "hip::std::numeric_limits<A>::is_exact == false");
    static_assert(hip::std::numeric_limits<A>::radix == 0,
                 "hip::std::numeric_limits<A>::radix == 0");
    assert(hip::std::numeric_limits<A>::epsilon() == A());
    assert(hip::std::numeric_limits<A>::round_error() == A());
    static_assert(hip::std::numeric_limits<A>::min_exponent == 0,
                 "hip::std::numeric_limits<A>::min_exponent == 0");
    static_assert(hip::std::numeric_limits<A>::min_exponent10 == 0,
                 "hip::std::numeric_limits<A>::min_exponent10 == 0");
    static_assert(hip::std::numeric_limits<A>::max_exponent == 0,
                 "hip::std::numeric_limits<A>::max_exponent == 0");
    static_assert(hip::std::numeric_limits<A>::max_exponent10 == 0,
                 "hip::std::numeric_limits<A>::max_exponent10 == 0");
    static_assert(hip::std::numeric_limits<A>::has_infinity == false,
                 "hip::std::numeric_limits<A>::has_infinity == false");
    static_assert(hip::std::numeric_limits<A>::has_quiet_NaN == false,
                 "hip::std::numeric_limits<A>::has_quiet_NaN == false");
    static_assert(hip::std::numeric_limits<A>::has_signaling_NaN == false,
                 "hip::std::numeric_limits<A>::has_signaling_NaN == false");
    static_assert(hip::std::numeric_limits<A>::has_denorm == hip::std::denorm_absent,
                 "hip::std::numeric_limits<A>::has_denorm == hip::std::denorm_absent");
    static_assert(hip::std::numeric_limits<A>::has_denorm_loss == false,
                 "hip::std::numeric_limits<A>::has_denorm_loss == false");
    assert(hip::std::numeric_limits<A>::infinity() == A());
    assert(hip::std::numeric_limits<A>::quiet_NaN() == A());
    assert(hip::std::numeric_limits<A>::signaling_NaN() == A());
    assert(hip::std::numeric_limits<A>::denorm_min() == A());
    static_assert(hip::std::numeric_limits<A>::is_iec559 == false,
                 "hip::std::numeric_limits<A>::is_iec559 == false");
    static_assert(hip::std::numeric_limits<A>::is_bounded == false,
                 "hip::std::numeric_limits<A>::is_bounded == false");
    static_assert(hip::std::numeric_limits<A>::is_modulo == false,
                 "hip::std::numeric_limits<A>::is_modulo == false");
    static_assert(hip::std::numeric_limits<A>::traps == false,
                 "hip::std::numeric_limits<A>::traps == false");
    static_assert(hip::std::numeric_limits<A>::tinyness_before == false,
                 "hip::std::numeric_limits<A>::tinyness_before == false");
    static_assert(hip::std::numeric_limits<A>::round_style == hip::std::round_toward_zero,
                 "hip::std::numeric_limits<A>::round_style == hip::std::round_toward_zero");

  return 0;
}
