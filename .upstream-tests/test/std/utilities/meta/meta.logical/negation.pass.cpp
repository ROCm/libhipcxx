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
// type_traits

// template<class B> struct negation;                        // C++17
// template<class B>
//   constexpr bool negation_v = negation<B>::value;         // C++17

#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

struct True  { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };

int main(int, char**)
{
    static_assert (!hip::std::negation<hip::std::true_type >::value, "" );
    static_assert ( hip::std::negation<hip::std::false_type>::value, "" );

    static_assert (!hip::std::negation_v<hip::std::true_type >, "" );
    static_assert ( hip::std::negation_v<hip::std::false_type>, "" );

    static_assert (!hip::std::negation<True >::value, "" );
    static_assert ( hip::std::negation<False>::value, "" );

    static_assert (!hip::std::negation_v<True >, "" );
    static_assert ( hip::std::negation_v<False>, "" );

    static_assert ( hip::std::negation<hip::std::negation<hip::std::true_type >>::value, "" );
    static_assert (!hip::std::negation<hip::std::negation<hip::std::false_type>>::value, "" );

  return 0;
}
