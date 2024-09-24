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

// type_traits

// integral_constant

#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef hip::std::integral_constant<int, 5> _5;
    static_assert(_5::value == 5, "");
    static_assert((hip::std::is_same<_5::value_type, int>::value), "");
    static_assert((hip::std::is_same<_5::type, _5>::value), "");
#if TEST_STD_VER >= 11
    static_assert((_5() == 5), "");
#endif
    assert(_5() == 5);


#if TEST_STD_VER > 11
    static_assert ( _5{}() == 5, "" );
    static_assert ( hip::std::true_type{}(), "" );
#endif

    static_assert(hip::std::false_type::value == false, "");
    static_assert((hip::std::is_same<hip::std::false_type::value_type, bool>::value), "");
    static_assert((hip::std::is_same<hip::std::false_type::type, hip::std::false_type>::value), "");

    static_assert(hip::std::true_type::value == true, "");
    static_assert((hip::std::is_same<hip::std::true_type::value_type, bool>::value), "");
    static_assert((hip::std::is_same<hip::std::true_type::type, hip::std::true_type>::value), "");

    hip::std::false_type f1;
    hip::std::false_type f2 = f1;
    assert(!f2);

    hip::std::true_type t1;
    hip::std::true_type t2 = t1;
    assert(t2);

  return 0;
}
