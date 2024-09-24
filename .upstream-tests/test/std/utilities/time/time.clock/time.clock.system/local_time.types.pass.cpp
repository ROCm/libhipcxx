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
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <cuda/std/chrono>

// struct local_t {};
// template<class Duration>
//   using local_time  = time_point<system_clock, Duration>;
// using local_seconds = sys_time<seconds>;
// using local_days    = sys_time<days>;

// [Example: 
//   sys_seconds{sys_days{1970y/January/1}}.time_since_epoch() is 0s. 
//   sys_seconds{sys_days{2000y/January/1}}.time_since_epoch() is 946’684’800s, which is 10’957 * 86’400s.
// —end example]


#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
#ifndef __cuda_std__
    using local_t = hip::std::chrono::local_t;
    using year    = hip::std::chrono::year;

    using seconds = hip::std::chrono::seconds;
    using minutes = hip::std::chrono::minutes;
    using days    = hip::std::chrono::days;
    
    using local_seconds = hip::std::chrono::local_seconds;
    using local_minutes = hip::std::chrono::local_time<minutes>;
    using local_days    = hip::std::chrono::local_days;

    constexpr hip::std::chrono::month January = hip::std::chrono::January;

    ASSERT_SAME_TYPE(hip::std::chrono::local_time<seconds>, local_seconds);
    ASSERT_SAME_TYPE(hip::std::chrono::local_time<days>,    local_days);

//  Test the long form, too
    ASSERT_SAME_TYPE(hip::std::chrono::time_point<local_t, seconds>, local_seconds);
    ASSERT_SAME_TYPE(hip::std::chrono::time_point<local_t, minutes>, local_minutes);
    ASSERT_SAME_TYPE(hip::std::chrono::time_point<local_t, days>,    local_days);
    
//  Test some well known values
    local_days d0 = local_days{year{1970}/January/1};
    local_days d1 = local_days{year{2000}/January/1};
    ASSERT_SAME_TYPE(decltype(d0.time_since_epoch()), days);
    assert( d0.time_since_epoch().count() == 0);
    assert( d1.time_since_epoch().count() == 10957);

    local_seconds s0{d0};
    local_seconds s1{d1};
    ASSERT_SAME_TYPE(decltype(s0.time_since_epoch()), seconds);
    assert( s0.time_since_epoch().count() == 0);
    assert( s1.time_since_epoch().count() == 946684800L);
#endif // __cuda_std__

    return 0;
}
