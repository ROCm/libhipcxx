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

// less_equal

#include <hip/std/functional>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#ifndef __CUDACC_RTC__
#include "pointer_comparison_test_helper.hpp"
#endif

int main(int, char**)
{
    typedef hip::std::less_equal<int> F;
    const F f = F();
#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    static_assert((hip::std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((hip::std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((hip::std::is_same<bool, F::result_type>::value), "" );
#endif
    assert(f(36, 36));
    assert(!f(36, 6));
    assert(f(6, 36));
#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    {
        // test total ordering of int* for less_equal<int*> and
        // less_equal<void>.
        do_pointer_comparison_test<int, hip::std::less_equal>();
    }
#endif
#if TEST_STD_VER > 11
    typedef hip::std::less_equal<> F2;
    const F2 f2 = F2();
    assert( f2(36, 36));
    assert(!f2(36, 6));
    assert( f2(6, 36));
    assert(!f2(36, 6.0));
    assert(!f2(36.0, 6));
    assert( f2(6, 36.0));
    assert( f2(6.0, 36));

    constexpr bool foo = hip::std::less_equal<int> () (36, 36);
    static_assert ( foo, "" );

    constexpr bool bar = hip::std::less_equal<> () (36.0, 36);
    static_assert ( bar, "" );
#endif

  return 0;
}
