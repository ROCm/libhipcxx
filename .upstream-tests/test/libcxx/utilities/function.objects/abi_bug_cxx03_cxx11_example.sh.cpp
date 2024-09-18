// -*- C++ -*-
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
// -*- C++ -*-

// REQUIRES: clang
// XFAIL: *

// This tests is meant to demonstrate an existing ABI bug between the
// C++03 and C++11 implementations of hip::std::function. It is not a real test.

// RUN: %cxx -c %s -o %t.first.o %flags %compile_flags -std=c++03 -g
// RUN: %cxx -c %s -o %t.second.o -DWITH_MAIN %flags %compile_flags -g -std=c++11
// RUN: %cxx -o %t.exe %t.first.o %t.second.o %flags %link_flags -g
// RUN: %run

#include <hip/std/functional>
#include <hip/std/cassert>

typedef hip::std::function<void(int)> Func;

Func CreateFunc();

#ifndef WITH_MAIN
// In C++03, the functions call operator, which is a part of the vtable,
// is defined as 'void operator()(int)', but in C++11 it's
// void operator()(int&&)'. So when the C++03 version is passed to C++11 code
// the value of the integer is interpreted as its address.
void test(int x) {
  assert(x == 42);
}
Func CreateFunc() {
  Func f(&test);
  return f;
}
#else
int main() {
  Func f = CreateFunc();
  f(42);
}
#endif
