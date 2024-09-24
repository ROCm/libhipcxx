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

// template<class... B> struct disjunction;                           // C++17
// template<class... B>
//   constexpr bool disjunction_v = disjunction<B...>::value;         // C++17

#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

struct True  { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };

int main(int, char**)
{
    static_assert (!hip::std::disjunction<>::value, "" );
    static_assert ( hip::std::disjunction<hip::std::true_type >::value, "" );
    static_assert (!hip::std::disjunction<hip::std::false_type>::value, "" );

    static_assert (!hip::std::disjunction_v<>, "" );
    static_assert ( hip::std::disjunction_v<hip::std::true_type >, "" );
    static_assert (!hip::std::disjunction_v<hip::std::false_type>, "" );

    static_assert ( hip::std::disjunction<hip::std::true_type,  hip::std::true_type >::value, "" );
    static_assert ( hip::std::disjunction<hip::std::true_type,  hip::std::false_type>::value, "" );
    static_assert ( hip::std::disjunction<hip::std::false_type, hip::std::true_type >::value, "" );
    static_assert (!hip::std::disjunction<hip::std::false_type, hip::std::false_type>::value, "" );

    static_assert ( hip::std::disjunction_v<hip::std::true_type,  hip::std::true_type >, "" );
    static_assert ( hip::std::disjunction_v<hip::std::true_type,  hip::std::false_type>, "" );
    static_assert ( hip::std::disjunction_v<hip::std::false_type, hip::std::true_type >, "" );
    static_assert (!hip::std::disjunction_v<hip::std::false_type, hip::std::false_type>, "" );

    static_assert ( hip::std::disjunction<hip::std::true_type,  hip::std::true_type,  hip::std::true_type >::value, "" );
    static_assert ( hip::std::disjunction<hip::std::true_type,  hip::std::false_type, hip::std::true_type >::value, "" );
    static_assert ( hip::std::disjunction<hip::std::false_type, hip::std::true_type,  hip::std::true_type >::value, "" );
    static_assert ( hip::std::disjunction<hip::std::false_type, hip::std::false_type, hip::std::true_type >::value, "" );
    static_assert ( hip::std::disjunction<hip::std::true_type,  hip::std::true_type,  hip::std::false_type>::value, "" );
    static_assert ( hip::std::disjunction<hip::std::true_type,  hip::std::false_type, hip::std::false_type>::value, "" );
    static_assert ( hip::std::disjunction<hip::std::false_type, hip::std::true_type,  hip::std::false_type>::value, "" );
    static_assert (!hip::std::disjunction<hip::std::false_type, hip::std::false_type, hip::std::false_type>::value, "" );

    static_assert ( hip::std::disjunction_v<hip::std::true_type,  hip::std::true_type,  hip::std::true_type >, "" );
    static_assert ( hip::std::disjunction_v<hip::std::true_type,  hip::std::false_type, hip::std::true_type >, "" );
    static_assert ( hip::std::disjunction_v<hip::std::false_type, hip::std::true_type,  hip::std::true_type >, "" );
    static_assert ( hip::std::disjunction_v<hip::std::false_type, hip::std::false_type, hip::std::true_type >, "" );
    static_assert ( hip::std::disjunction_v<hip::std::true_type,  hip::std::true_type,  hip::std::false_type>, "" );
    static_assert ( hip::std::disjunction_v<hip::std::true_type,  hip::std::false_type, hip::std::false_type>, "" );
    static_assert ( hip::std::disjunction_v<hip::std::false_type, hip::std::true_type,  hip::std::false_type>, "" );
    static_assert (!hip::std::disjunction_v<hip::std::false_type, hip::std::false_type, hip::std::false_type>, "" );

    static_assert ( hip::std::disjunction<True >::value, "" );
    static_assert (!hip::std::disjunction<False>::value, "" );

    static_assert ( hip::std::disjunction_v<True >, "" );
    static_assert (!hip::std::disjunction_v<False>, "" );

  return 0;
}
