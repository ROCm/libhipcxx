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

// type_traits

// make_signed

#include <hip/std/type_traits>

#include "test_macros.h"

enum Enum {zero, one_};

#if TEST_STD_VER >= 11
enum BigEnum : unsigned long long // MSVC's ABI doesn't follow the Standard
#else
enum BigEnum
#endif
{
    bigzero,
    big = 0xFFFFFFFFFFFFFFFFULL
};

#if !defined(_LIBCUDACXX_HAS_NO_INT128) && !defined(_LIBCUDACXX_HAS_NO_STRONG_ENUMS)
enum HugeEnum : __uint128_t
{
    hugezero
};
#endif

template <class T, class U>
__host__ __device__
void test_make_signed()
{
    ASSERT_SAME_TYPE(U, typename hip::std::make_signed<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U, hip::std::make_signed_t<T>);
#endif
}

int main(int, char**)
{
    test_make_signed< signed char, signed char >();
    test_make_signed< unsigned char, signed char >();
    test_make_signed< char, signed char >();
    test_make_signed< short, signed short >();
    test_make_signed< unsigned short, signed short >();
    test_make_signed< int, signed int >();
    test_make_signed< unsigned int, signed int >();
    test_make_signed< long, signed long >();
    test_make_signed< unsigned long, long >();
    test_make_signed< long long, signed long long >();
    test_make_signed< unsigned long long, signed long long >();
    test_make_signed< wchar_t, hip::std::conditional<sizeof(wchar_t) == 4, int, short>::type >();
    test_make_signed< const wchar_t, hip::std::conditional<sizeof(wchar_t) == 4, const int, const short>::type >();
    test_make_signed< const Enum, hip::std::conditional<sizeof(Enum) == sizeof(int), const int, const signed char>::type >();
    test_make_signed< BigEnum, hip::std::conditional<sizeof(long) == 4, long long, long>::type >();
#ifndef _LIBCUDACXX_HAS_NO_INT128
    test_make_signed< __int128_t, __int128_t >();
    test_make_signed< __uint128_t, __int128_t >();
# ifndef _LIBCUDACXX_HAS_NO_STRONG_ENUMS
    test_make_signed< HugeEnum, __int128_t >();
# endif
#endif

  return 0;
}
