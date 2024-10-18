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

#include <hip/std/tuple>
#include <hip/std/string>
// UNSUPPORTED: hipcc
// Marking the test as unsupported:
// This test is including hip/std/string which does not exist.
// HIPCC gives the correct error: hip/std/string file not found.
// However, other errors are expected, consider the expected-error and expected-note. 
// We found _LIBHIPCXX_HAS_STRING guard in other tests that include the header, 
// indicating that the header is not implemented yet. 

#include "test_macros.h"

struct UserType {};

void test_bad_index() {
    hip::std::tuple<long, long, char, hip::std::string, char, UserType, char> t1;
    TEST_IGNORE_NODISCARD hip::std::get<int>(t1); // expected-error@tuple:* {{type not found}}
    TEST_IGNORE_NODISCARD hip::std::get<long>(t1); // expected-note {{requested here}}
    TEST_IGNORE_NODISCARD hip::std::get<char>(t1); // expected-note {{requested here}}
        // expected-error@tuple:* 2 {{type occurs more than once}}
    hip::std::tuple<> t0;
    TEST_IGNORE_NODISCARD hip::std::get<char*>(t0); // expected-node {{requested here}}
        // expected-error@tuple:* 1 {{type not in empty type list}}
}

void test_bad_return_type() {
    typedef hip::std::unique_ptr<int> upint;
    hip::std::tuple<upint> t;
    upint p = hip::std::get<upint>(t); // expected-error{{deleted copy constructor}}
}

int main(int, char**)
{
    test_bad_index();
    test_bad_return_type();

  return 0;
}