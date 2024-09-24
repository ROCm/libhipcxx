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
//
// Test the "test_macros.h" header.
#include <hip/std/detail/__config>
#include "test_macros.h"

#ifndef TEST_STD_VER
#error TEST_STD_VER must be defined
#endif

#ifndef TEST_NOEXCEPT
#error TEST_NOEXCEPT must be defined
#endif

#ifndef LIBCPP_ASSERT
#error LIBCPP_ASSERT must be defined
#endif

#ifndef LIBCPP_STATIC_ASSERT
#error LIBCPP_STATIC_ASSERT must be defined
#endif

__host__ __device__
void test_noexcept() TEST_NOEXCEPT
{
}

__host__ __device__
void test_libcxx_macros()
{
//  ===== C++14 features =====
//  defined(TEST_HAS_EXTENDED_CONSTEXPR)  != defined(_LIBCUDACXX_HAS_NO_CXX14_CONSTEXPR)
#ifdef TEST_HAS_EXTENDED_CONSTEXPR
# ifdef _LIBCUDACXX_HAS_NO_CXX14_CONSTEXPR
#  error "TEST_EXTENDED_CONSTEXPR mismatch (1)"
# endif
#else
# ifndef _LIBCUDACXX_HAS_NO_CXX14_CONSTEXPR
#  error "TEST_EXTENDED_CONSTEXPR mismatch (2)"
# endif
#endif

//  defined(TEST_HAS_VARIABLE_TEMPLATES) != defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
#ifdef TEST_HAS_VARIABLE_TEMPLATES
# ifdef _LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES
#  error "TEST_VARIABLE_TEMPLATES mismatch (1)"
# endif
#else
# ifndef _LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES
#  error "TEST_VARIABLE_TEMPLATES mismatch (2)"
# endif
#endif

//  ===== C++17 features =====
}

int main(int, char**)
{
    test_noexcept();
    test_libcxx_macros();

  return 0;
}
