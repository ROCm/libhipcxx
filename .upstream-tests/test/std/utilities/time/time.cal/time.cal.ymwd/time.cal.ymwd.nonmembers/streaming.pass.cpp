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
// UNSUPPORTED: c++98, c++03, c++11, nvrtc
// XFAIL: *

// <chrono>
// class year_month_weekday;

// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const year_month_weekday& ym);
//
// Returns: os << ym.year() << '/' << ym.month().
//
//
// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     to_stream(basic_ostream<charT, traits>& os, const charT* fmt, const year_month_weekday& ym);
//
// Effects: Streams ym into os using the format specified by the NTCTS fmt. fmt encoding follows the rules specified in 25.11.
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//     basic_istream<charT, traits>&
//   from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//               year_month_weekday& ym, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//               minutes* offset = nullptr);
//
// Effects: Attempts to parse the input stream is into the year_month_weekday ym using the format
//         flags given in the NTCTS fmt as specified in 25.12. If the parse fails to decode
//         a valid year_month_weekday, is.setstate(ios_- base::failbit) shall be called and ym shall
//         not be modified. If %Z is used and successfully parsed, that value will be assigned
//         to *abbrev if abbrev is non-null. If %z (or a modified variant) is used and
//         successfully parsed, that value will be assigned to *offset if offset is non-null.



#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>
#include <iostream>
#include "test_macros.h"

int main(int, char**)
{
    using year_month_weekday = hip::std::chrono::year_month_weekday;
    using year                = hip::std::chrono::year;
    using month               = hip::std::chrono::month;
    using weekday             = hip::std::chrono::weekday;

    std::cout << year_month_weekday{year{2018}, month{3}, weekday{4}};

  return 0;
}
