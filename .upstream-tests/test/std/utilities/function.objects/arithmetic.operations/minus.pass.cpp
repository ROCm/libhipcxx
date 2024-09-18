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

// <cuda/std/functional>

// minus

#include <hip/std/functional>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef hip::std::minus<int> F;
    const F f = F();
#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    static_assert((hip::std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((hip::std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((hip::std::is_same<int, F::result_type>::value), "" );
#endif
    assert(f(3, 2) == 1);
#if TEST_STD_VER > 11
    typedef hip::std::minus<> F2;
    const F2 f2 = F2();
    assert(f2(3,2) == 1);
    assert(f2(3.0, 2) == 1);
    assert(f2(3, 2.5) == 0.5);

    constexpr int foo = hip::std::minus<int> () (3, 2);
    static_assert ( foo == 1, "" );

    constexpr double bar = hip::std::minus<> () (3.0, 2);
    static_assert ( bar == 1.0, "" );
#endif

  return 0;
}
