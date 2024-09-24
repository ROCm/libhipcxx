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
#include <hip/std/type_traits>
#include "test_macros.h"

// XFAIL: c++98, c++03, c++11

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

// hip::std::byte is not an integer type, nor a character type.
// It is a distinct type for accessing the bits that ultimately make up object storage.

#if TEST_STD_VER > 11
static_assert( hip::std::is_trivial<hip::std::byte>::value, "" );   // P0767
#else
static_assert( hip::std::is_pod<hip::std::byte>::value, "" );
#endif
static_assert(!hip::std::is_arithmetic<hip::std::byte>::value, "" );
static_assert(!hip::std::is_integral<hip::std::byte>::value, "" );

static_assert(!hip::std::is_same<hip::std::byte,          char>::value, "" );
static_assert(!hip::std::is_same<hip::std::byte,   signed char>::value, "" );
static_assert(!hip::std::is_same<hip::std::byte, unsigned char>::value, "" );

// The standard doesn't outright say this, but it's pretty clear that it has to be true.
static_assert(sizeof(hip::std::byte) == 1, "" );

int main(int, char**) {
  return 0;
}
