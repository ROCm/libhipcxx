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

// bit_or

#include <hip/std/functional>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef hip::std::bit_or<int> F;
    const F f = F();
#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    static_assert((hip::std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((hip::std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((hip::std::is_same<int, F::result_type>::value), "" );
#endif 
    assert(f(0xEA95, 0xEA95) == 0xEA95);
    assert(f(0xEA95, 0x58D3) == 0xFAD7);
    assert(f(0x58D3, 0xEA95) == 0xFAD7);
    assert(f(0x58D3, 0) == 0x58D3);
    assert(f(0xFFFF, 0x58D3) == 0xFFFF);
#if TEST_STD_VER > 11
    typedef hip::std::bit_or<> F2;
    const F2 f2 = F2();
    assert(f2(0xEA95, 0xEA95) == 0xEA95);
    assert(f2(0xEA95L, 0xEA95) == 0xEA95);
    assert(f2(0xEA95, 0xEA95L) == 0xEA95);

    assert(f2(0xEA95, 0x58D3) == 0xFAD7);
    assert(f2(0xEA95L, 0x58D3) == 0xFAD7);
    assert(f2(0xEA95, 0x58D3L) == 0xFAD7);

    assert(f2(0x58D3, 0xEA95) == 0xFAD7);
    assert(f2(0x58D3L, 0xEA95) == 0xFAD7);
    assert(f2(0x58D3, 0xEA95L) == 0xFAD7);

    assert(f2(0x58D3, 0) == 0x58D3);
    assert(f2(0x58D3L, 0) == 0x58D3);
    assert(f2(0x58D3, 0L) == 0x58D3);

    assert(f2(0xFFFF, 0x58D3) == 0xFFFF);
    assert(f2(0xFFFFL, 0x58D3) == 0xFFFF);
    assert(f2(0xFFFF, 0x58D3L) == 0xFFFF);

    constexpr int foo = hip::std::bit_or<int> () (0x58D3, 0xEA95);
    static_assert ( foo == 0xFAD7, "" );

    constexpr int bar = hip::std::bit_or<> () (0x58D3L, 0xEA95);
    static_assert ( bar == 0xFAD7, "" );
#endif

  return 0;
}
