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

#pragma nv_diag_suppress declared_but_not_referenced
#pragma nv_diag_suppress set_but_not_used

#define _LIBHIPCXX_HIP_ABI_VERSION 3

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

int main(int, char**)
{
    using namespace hip::std::literals::chrono_literals;

// long long ABI v3 check
  {
    constexpr auto _h   = 3h;
    constexpr auto _min = 3min;
    constexpr auto _s   = 3s;
    constexpr auto _ms  = 3ms;
    constexpr auto _us  = 3us;
    constexpr auto _ns  = 3ns;

    unused(_h);
    unused(_min);
    unused(_s);
    unused(_ms);
    unused(_us);
    unused(_ns);

    static_assert(hip::std::is_same< decltype(_h.count()),   hip::std::chrono::hours::rep        >::value, "");
    static_assert(hip::std::is_same< decltype(_min.count()), hip::std::chrono::minutes::rep      >::value, "");
    static_assert(hip::std::is_same< decltype(_s.count()),   hip::std::chrono::seconds::rep      >::value, "");
    static_assert(hip::std::is_same< decltype(_ms.count()),  hip::std::chrono::milliseconds::rep >::value, "");
    static_assert(hip::std::is_same< decltype(_us.count()),  hip::std::chrono::microseconds::rep >::value, "");
    static_assert(hip::std::is_same< decltype(_ns.count()),  hip::std::chrono::nanoseconds::rep  >::value, "");

    static_assert ( hip::std::is_same<decltype(3h), hip::std::chrono::hours>::value, "" );
    static_assert ( hip::std::is_same<decltype(3min), hip::std::chrono::minutes>::value, "" );
    static_assert ( hip::std::is_same<decltype(3s), hip::std::chrono::seconds>::value, "" );
    static_assert ( hip::std::is_same<decltype(3ms), hip::std::chrono::milliseconds>::value, "" );
    static_assert ( hip::std::is_same<decltype(3us), hip::std::chrono::microseconds>::value, "" );
    static_assert ( hip::std::is_same<decltype(3ns), hip::std::chrono::nanoseconds>::value, "" );
  }

// long double ABI v3 check
  {
    constexpr auto _h   = 3.0h;
    constexpr auto _min = 3.0min;
    constexpr auto _s   = 3.0s;
    constexpr auto _ms  = 3.0ms;
    constexpr auto _us  = 3.0us;
    constexpr auto _ns  = 3.0ns;

    unused(_h);
    unused(_min);
    unused(_s);
    unused(_ms);
    unused(_us);
    unused(_ns);

    using hip::std::ratio;
    using hip::std::milli;
    using hip::std::micro;
    using hip::std::nano;

    static_assert(hip::std::is_same< decltype(_h.count()),   hip::std::chrono::duration<long double, ratio<3600>>::rep        >::value, "");
    static_assert(hip::std::is_same< decltype(_min.count()), hip::std::chrono::duration<long double, ratio<  60>>::rep      >::value, "");
    // static_assert(hip::std::is_same< decltype(s.count()),   hip::std::chrono::duration<long double             >::rep      >::value, "");
    static_assert(hip::std::is_same< decltype(_ms.count()),  hip::std::chrono::duration<long double,       milli>::rep >::value, "");
    static_assert(hip::std::is_same< decltype(_us.count()),  hip::std::chrono::duration<long double,       micro>::rep >::value, "");
    static_assert(hip::std::is_same< decltype(_ns.count()),  hip::std::chrono::duration<long double,        nano>::rep  >::value, "");
  }

  return 0;
}
