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

// unary_negate

#define _LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS
#define _LIBCUDACXX_ENABLE_CXX20_REMOVED_NEGATORS
#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <hip/std/functional>
#include <hip/std/type_traits>
#include <hip/std/cassert>

int main(int, char**)
{
    typedef hip::std::unary_negate<hip::std::logical_not<int> > F;
    const F f = F(hip::std::logical_not<int>());
#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    static_assert((hip::std::is_same<F::argument_type, int>::value), "" );
    static_assert((hip::std::is_same<F::result_type, bool>::value), "" );
#endif
    assert(f(36));
    assert(!f(0));

  return 0;
}
