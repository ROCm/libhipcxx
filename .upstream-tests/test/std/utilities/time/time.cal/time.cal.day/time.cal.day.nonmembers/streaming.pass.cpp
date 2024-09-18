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
// class day;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& os, const day& d);
//
//   Effects: Inserts format(fmt, d) where fmt is "%d" widened to charT.
//                If !d.ok(), appends with " is not a valid day".
//
// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//   to_stream(basic_ostream<charT, traits>& os, const charT* fmt, const day& d);
//
//   Effects: Streams d into os using the format specified by the NTCTS fmt.
//              fmt encoding follows the rules specified in 25.11.
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//   from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//             day& d, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//             minutes* offset = nullptr);
//
//   Effects: Attempts to parse the input stream is into the day d using the format flags
//             given in the NTCTS fmt as specified in 25.12.
//             If the parse fails to decode a valid day, is.setstate(ios_base::failbit)
//             shall be called and d shall not be modified.
//             If %Z is used and successfully parsed, that value will be assigned to *abbrev
//             if abbrev is non-null. If %z (or a modified variant) is used and
//             successfully parsed, that value will be assigned to *offset if offset is non-null.
//


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>
#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
   using day = hip::std::chrono::day;
   std::cout << day{1};

  return 0;
}
