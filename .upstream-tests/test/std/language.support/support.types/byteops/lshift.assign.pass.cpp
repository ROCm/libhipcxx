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

#include <hip/std/cstddef>
#include <test_macros.h>

// UNSUPPORTED: c++98, c++03, c++11, c++14

// template <class IntegerType>
//   constexpr byte& operator<<=(byte& b, IntegerType shift) noexcept;
// This function shall not participate in overload resolution unless
//   is_integral_v<IntegerType> is true.


__host__ __device__
constexpr hip::std::byte test(hip::std::byte b) {
    return b <<= 2;
    }


int main(int, char**) {
    hip::std::byte b;  // not constexpr, just used in noexcept check
    constexpr hip::std::byte b2{static_cast<hip::std::byte>(2)};
    constexpr hip::std::byte b3{static_cast<hip::std::byte>(3)};

    static_assert(noexcept(b <<= 2), "" );

    static_assert(hip::std::to_integer<int>(test(b2)) ==  8, "" );
    static_assert(hip::std::to_integer<int>(test(b3)) == 12, "" );


  return 0;
}
