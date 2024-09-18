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
// This is a dummy feature that prevents this test from running by default.
// REQUIRES: template-const-testing

// The table below compares the compile time and object size for each of the
// variants listed in the RUN script.
//
//  Impl          Compile Time    Object Size
// -------------------------------------------
// _And:         3,498.639 ms     158 M
// __lazy_and:  10,138.982 ms     334 M
// __and_:      14,181.851 ms     648 M
//

// RUN: %cxx %flags %compile_flags -c %s -o %S/new.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace -std=c++17
// RUN: %cxx %flags %compile_flags -c %s -o %S/lazy.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace  -std=c++17 -DTEST_LAZY_AND
// RUN: %cxx %flags %compile_flags -c %s -o %S/std.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace   -std=c++17 -DTEST_STD_AND

#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "template_cost_testing.h"
using hip::std::true_type;
using hip::std::false_type;

#define FALSE_T() hip::std::false_type,
#define TRUE_T() hip::std::true_type,

#ifdef TEST_LAZY_AND
#define TEST_AND hip::std::__lazy_and
#define TEST_OR hip::std::__lazy_or
#elif defined(TEST_STD_AND)
#define TEST_AND hip::std::__and_
#define TEST_OR hip::std::__or_
#else
#define TEST_AND hip::std::_And
#define TEST_OR hip::std::_Or
#endif

void sink(...);

void Foo1(TEST_AND < REPEAT_1000(TRUE_T) true_type > t1) { sink(&t1); }
void Foo2(TEST_AND < REPEAT_1000(TRUE_T) REPEAT_1000(TRUE_T) true_type > t2) { sink(&t2); }
void Foo3(TEST_AND < REPEAT_1000(TRUE_T) true_type, false_type > t3) { sink(&t3); }
void Foo4(TEST_AND < REPEAT_1000(TRUE_T) REPEAT_1000(TRUE_T) true_type, false_type > t4) { sink(&t4); }
void Foo5(TEST_AND < false_type, REPEAT_1000(TRUE_T) true_type > t5) { sink(&t5); }
void Foo6(TEST_AND < false_type, REPEAT_1000(TRUE_T) REPEAT_1000(TRUE_T) true_type > t6) { sink(&t6); }

void escape() {

sink(&Foo1);
sink(&Foo2);
sink(&Foo3);
sink(&Foo4);
sink(&Foo5);
sink(&Foo6);
}


