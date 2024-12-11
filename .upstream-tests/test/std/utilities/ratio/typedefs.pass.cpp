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

// test ratio typedef's

#include <hip/std/ratio>

int main(int, char**)
{
    static_assert(hip::std::atto::num == 1 && hip::std::atto::den == 1000000000000000000ULL, "");
    static_assert(hip::std::femto::num == 1 && hip::std::femto::den == 1000000000000000ULL, "");
    static_assert(hip::std::pico::num == 1 && hip::std::pico::den == 1000000000000ULL, "");
    static_assert(hip::std::nano::num == 1 && hip::std::nano::den == 1000000000ULL, "");
    static_assert(hip::std::micro::num == 1 && hip::std::micro::den == 1000000ULL, "");
    static_assert(hip::std::milli::num == 1 && hip::std::milli::den == 1000ULL, "");
    static_assert(hip::std::centi::num == 1 && hip::std::centi::den == 100ULL, "");
    static_assert(hip::std::deci::num == 1 && hip::std::deci::den == 10ULL, "");
    static_assert(hip::std::deca::num == 10ULL && hip::std::deca::den == 1, "");
    static_assert(hip::std::hecto::num == 100ULL && hip::std::hecto::den == 1, "");
    static_assert(hip::std::kilo::num == 1000ULL && hip::std::kilo::den == 1, "");
    static_assert(hip::std::mega::num == 1000000ULL && hip::std::mega::den == 1, "");
    static_assert(hip::std::giga::num == 1000000000ULL && hip::std::giga::den == 1, "");
    static_assert(hip::std::tera::num == 1000000000000ULL && hip::std::tera::den == 1, "");
    static_assert(hip::std::peta::num == 1000000000000000ULL && hip::std::peta::den == 1, "");
    static_assert(hip::std::exa::num == 1000000000000000000ULL && hip::std::exa::den == 1, "");

  return 0;
}
