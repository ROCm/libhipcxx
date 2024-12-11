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
// <cuda/std/functional>

// bit_not

#include <hip/std/functional>
#include <hip/std/type_traits>
#include <hip/std/cassert>

int main(int, char**)
{
    typedef hip::std::bit_not<int> F;
    const F f = F();
#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    static_assert((hip::std::is_same<F::argument_type, int>::value), "" );
    static_assert((hip::std::is_same<F::result_type, int>::value), "" );
#endif
    assert((f(0xEA95) & 0xFFFF ) == 0x156A);
    assert((f(0x58D3) & 0xFFFF ) == 0xA72C);
    assert((f(0)      & 0xFFFF ) == 0xFFFF);
    assert((f(0xFFFF) & 0xFFFF ) == 0);

    typedef hip::std::bit_not<> F2;
    const F2 f2 = F2();
    assert((f2(0xEA95)  & 0xFFFF ) == 0x156A);
    assert((f2(0xEA95L) & 0xFFFF ) == 0x156A);
    assert((f2(0x58D3)  & 0xFFFF ) == 0xA72C);
    assert((f2(0x58D3L) & 0xFFFF ) == 0xA72C);
    assert((f2(0)       & 0xFFFF ) == 0xFFFF);
    assert((f2(0L)      & 0xFFFF ) == 0xFFFF);
    assert((f2(0xFFFF)  & 0xFFFF ) == 0);
    assert((f2(0xFFFFL)  & 0xFFFF ) == 0);

    constexpr int foo = hip::std::bit_not<int> () (0xEA95) & 0xFFFF;
    static_assert ( foo == 0x156A, "" );

    constexpr int bar = hip::std::bit_not<> () (0xEA95) & 0xFFFF;
    static_assert ( bar == 0x156A, "" );

  return 0;
}
