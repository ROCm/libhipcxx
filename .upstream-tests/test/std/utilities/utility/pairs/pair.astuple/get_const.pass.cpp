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

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     const typename tuple_element<I, hip::std::pair<T1, T2> >::type&
//     get(const pair<T1, T2>&);

// UNSUPPORTED: msvc

#include <hip/std/utility>
#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef hip::std::pair<int, short> P;
        const P p(3, static_cast<short>(4));
        assert(hip::std::get<0>(p) == 3);
        assert(hip::std::get<1>(p) == 4);
    }

#if TEST_STD_VER > 11
    {
        typedef hip::std::pair<int, short> P;
        constexpr P p1(3, static_cast<short>(4));
        static_assert(hip::std::get<0>(p1) == 3, "");
        static_assert(hip::std::get<1>(p1) == 4, "");
    }
#endif

  return 0;
}
