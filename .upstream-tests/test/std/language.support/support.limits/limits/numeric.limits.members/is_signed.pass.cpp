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

// is_signed

#include <hip/std/limits>

#include "test_macros.h"

template <class T, bool expected>
__host__ __device__
void
test()
{
    static_assert(hip::std::numeric_limits<T>::is_signed == expected, "is_signed test 1");
    static_assert(hip::std::numeric_limits<const T>::is_signed == expected, "is_signed test 2");
    static_assert(hip::std::numeric_limits<volatile T>::is_signed == expected, "is_signed test 3");
    static_assert(hip::std::numeric_limits<const volatile T>::is_signed == expected, "is_signed test 4");
}

int main(int, char**)
{
    test<bool, false>();
    test<char, char(-1) < char(0)>();
    test<signed char, true>();
    test<unsigned char, false>();
    test<wchar_t, wchar_t(-1) < wchar_t(0)>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, false>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<char16_t, false>();
    test<char32_t, false>();
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<short, true>();
    test<unsigned short, false>();
    test<int, true>();
    test<unsigned int, false>();
    test<long, true>();
    test<unsigned long, false>();
    test<long long, true>();
    test<unsigned long long, false>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
    test<__int128_t, true>();
    test<__uint128_t, false>();
#endif
    test<float, true>();
    test<double, true>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
    test<long double, true>();
#endif

    return 0;
}
