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

// constexpr byte operator&(byte l, byte r) noexcept;

int main(int, char**) {
    constexpr hip::std::byte b1{static_cast<hip::std::byte>(1)};
    constexpr hip::std::byte b8{static_cast<hip::std::byte>(8)};
    constexpr hip::std::byte b9{static_cast<hip::std::byte>(9)};

    static_assert(noexcept(b1 & b8), "" );

    static_assert(hip::std::to_integer<int>(b1 & b8) ==  0, "");
    static_assert(hip::std::to_integer<int>(b1 & b9) ==  1, "");
    static_assert(hip::std::to_integer<int>(b8 & b9) ==  8, "");

    static_assert(hip::std::to_integer<int>(b8 & b1) ==  0, "");
    static_assert(hip::std::to_integer<int>(b9 & b1) ==  1, "");
    static_assert(hip::std::to_integer<int>(b9 & b8) ==  8, "");

  return 0;
}
