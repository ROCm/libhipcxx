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

// digits10

#include <hip/std/limits>
#include <hip/std/cfloat>

#include "test_macros.h"

template <class T, int expected>
__host__ __device__
void
test()
{
    static_assert(hip::std::numeric_limits<T>::digits10 == expected, "digits10 test 1");
    static_assert(hip::std::numeric_limits<T>::is_bounded, "digits10 test 5");
    static_assert(hip::std::numeric_limits<const T>::digits10 == expected, "digits10 test 2");
    static_assert(hip::std::numeric_limits<const T>::is_bounded, "digits10 test 6");
    static_assert(hip::std::numeric_limits<volatile T>::digits10 == expected, "digits10 test 3");
    static_assert(hip::std::numeric_limits<volatile T>::is_bounded, "digits10 test 7");
    static_assert(hip::std::numeric_limits<const volatile T>::digits10 == expected, "digits10 test 4");
    static_assert(hip::std::numeric_limits<const volatile T>::is_bounded, "digits10 test 8");
}

int main(int, char**)
{
    test<bool, 0>();
    test<char, 2>();
    test<signed char, 2>();
    test<unsigned char, 2>();
    test<wchar_t, 5*sizeof(wchar_t)/2-1>();  // 4 -> 9 and 2 -> 4
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, 2>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<char16_t, 4>();
    test<char32_t, 9>();
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<short, 4>();
    test<unsigned short, 4>();
    test<int, 9>();
    test<unsigned int, 9>();
    test<long, sizeof(long) == 4 ? 9 : 18>();
    test<unsigned long, sizeof(long) == 4 ? 9 : 19>();
    test<long long, 18>();
    test<unsigned long long, 19>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
    test<__int128_t, 38>();
    test<__uint128_t, 38>();
#endif
    test<float, FLT_DIG>();
    test<double, DBL_DIG>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
    test<long double, LDBL_DIG>();
#endif

    return 0;
}
