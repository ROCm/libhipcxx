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
//

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

//  Tuples of smart pointers; based on bug #18350
//  auto_ptr doesn't have a copy constructor that takes a const &, but tuple does.

#include <hip/std/tuple>
// hip::std::___ptr not supported
//#include <memory>

#include "test_macros.h"

int main(int, char**) {
  // hip::std::___ptr not supported
  /*
    {
    hip::std::tuple<hip::std::unique_ptr<char>> up;
    hip::std::tuple<hip::std::shared_ptr<char>> sp;
    hip::std::tuple<hip::std::weak_ptr  <char>> wp;
    }
    {
    hip::std::tuple<hip::std::unique_ptr<char[]>> up;
    hip::std::tuple<hip::std::shared_ptr<char[]>> sp;
    hip::std::tuple<hip::std::weak_ptr  <char[]>> wp;
    }
  */
    // Smart pointers of type 'T[N]' are not tested here since they are not
    // supported by the standard nor by libc++'s implementation.
    // See https://reviews.llvm.org/D21320 for more information.

  return 0;
}
