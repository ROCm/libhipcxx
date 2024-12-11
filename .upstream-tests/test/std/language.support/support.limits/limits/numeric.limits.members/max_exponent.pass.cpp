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

// max_exponent

#include <hip/std/limits>
#include <hip/std/cfloat>

#include "test_macros.h"

template <class T, int expected>
__host__ __device__
void
test()
{
    static_assert(hip::std::numeric_limits<T>::max_exponent == expected, "max_exponent test 1");
    static_assert(hip::std::numeric_limits<const T>::max_exponent == expected, "max_exponent test 2");
    static_assert(hip::std::numeric_limits<volatile T>::max_exponent == expected, "max_exponent test 3");
    static_assert(hip::std::numeric_limits<const volatile T>::max_exponent == expected, "max_exponent test 4");
}

int main(int, char**)
{
    test<bool, 0>();
    test<char, 0>();
    test<signed char, 0>();
    test<unsigned char, 0>();
    test<wchar_t, 0>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, 0>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<char16_t, 0>();
    test<char32_t, 0>();
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<short, 0>();
    test<unsigned short, 0>();
    test<int, 0>();
    test<unsigned int, 0>();
    test<long, 0>();
    test<unsigned long, 0>();
    test<long long, 0>();
    test<unsigned long long, 0>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
    test<__int128_t, 0>();
    test<__uint128_t, 0>();
#endif
    test<float, FLT_MAX_EXP>();
    test<double, DBL_MAX_EXP>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
    test<long double, LDBL_MAX_EXP>();
#endif

    return 0;
}
