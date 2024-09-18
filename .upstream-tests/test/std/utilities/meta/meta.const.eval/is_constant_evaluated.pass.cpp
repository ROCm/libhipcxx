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

// UNSUPPORTED: c++98, c++03

// <cuda/std/type_traits>

// constexpr bool is_constant_evaluated() noexcept; // C++20

#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

// libcudacxx does not have feature test macros... yet
/*
#ifndef __cpp_lib_is_constant_evaluated
#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
# error __cpp_lib_is_constant_evaluated should be defined
#endif
#endif
*/

template <bool> struct InTemplate {};

int main(int, char**)
{
#if defined(_LIBCUDACXX_IS_CONSTANT_EVALUATED)
  // Test the signature
  {
    ASSERT_SAME_TYPE(decltype(hip::std::is_constant_evaluated()), bool);
    ASSERT_NOEXCEPT(hip::std::is_constant_evaluated());
    constexpr bool p = hip::std::is_constant_evaluated();
    assert(p);
  }
  // Test the return value of the builtin for basic sanity only. It's the
  // compilers job to test tho builtin for correctness.
  {
    static_assert(hip::std::is_constant_evaluated(), "");
    bool p = hip::std::is_constant_evaluated();
    assert(!p);
    ASSERT_SAME_TYPE(InTemplate<hip::std::is_constant_evaluated()>, InTemplate<true>);
    static int local_static = hip::std::is_constant_evaluated() ? 42 : -1;
    assert(local_static == 42);
  }
#endif
  return 0;
}
