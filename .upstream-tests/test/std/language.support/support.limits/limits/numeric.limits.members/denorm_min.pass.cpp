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

// test numeric_limits

// denorm_min()

#include <hip/std/limits>
#include <hip/std/cfloat>
#include <hip/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__
void
test(T expected)
{
    assert(hip::std::numeric_limits<T>::denorm_min() == expected);
    assert(hip::std::numeric_limits<const T>::denorm_min() == expected);
    assert(hip::std::numeric_limits<volatile T>::denorm_min() == expected);
    assert(hip::std::numeric_limits<const volatile T>::denorm_min() == expected);
}

int main(int, char**)
{
    test<bool>(false);
    test<char>(0);
    test<signed char>(0);
    test<unsigned char>(0);
    test<wchar_t>(0);
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>(0);
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<char16_t>(0);
    test<char32_t>(0);
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<short>(0);
    test<unsigned short>(0);
    test<int>(0);
    test<unsigned int>(0);
    test<long>(0);
    test<unsigned long>(0);
    test<long long>(0);
    test<unsigned long long>(0);
#ifndef _LIBCUDACXX_HAS_NO_INT128
    test<__int128_t>(0);
    test<__uint128_t>(0);
#endif
#if defined(__FLT_DENORM_MIN__) // guarded because these macros are extensions.
    test<float>(__FLT_DENORM_MIN__);
    test<double>(__DBL_DENORM_MIN__);
    #ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
    test<long double>(__LDBL_DENORM_MIN__);
    #endif
#endif
#if defined(FLT_TRUE_MIN) // not currently provided on linux.
    test<float>(FLT_TRUE_MIN);
    test<double>(DBL_TRUE_MIN);
    #ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
    test<long double>(LDBL_TRUE_MIN);
    #endif
#endif
#if !defined(__FLT_DENORM_MIN__) && !defined(FLT_TRUE_MIN)
#error Test has no expected values for floating point types
#endif

  return 0;
}
