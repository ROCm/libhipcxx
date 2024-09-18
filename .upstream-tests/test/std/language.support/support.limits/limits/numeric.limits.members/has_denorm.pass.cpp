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

// has_denorm

#include <hip/std/limits>

#include "test_macros.h"

template <class T, hip::std::float_denorm_style expected>
__host__ __device__
void
test()
{
    static_assert(hip::std::numeric_limits<T>::has_denorm == expected, "has_denorm test 1");
    static_assert(hip::std::numeric_limits<const T>::has_denorm == expected, "has_denorm test 2");
    static_assert(hip::std::numeric_limits<volatile T>::has_denorm == expected, "has_denorm test 3");
    static_assert(hip::std::numeric_limits<const volatile T>::has_denorm == expected, "has_denorm test 4");
}

int main(int, char**)
{
    test<bool, hip::std::denorm_absent>();
    test<char, hip::std::denorm_absent>();
    test<signed char, hip::std::denorm_absent>();
    test<unsigned char, hip::std::denorm_absent>();
    test<wchar_t, hip::std::denorm_absent>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, hip::std::denorm_absent>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<char16_t, hip::std::denorm_absent>();
    test<char32_t, hip::std::denorm_absent>();
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<short, hip::std::denorm_absent>();
    test<unsigned short, hip::std::denorm_absent>();
    test<int, hip::std::denorm_absent>();
    test<unsigned int, hip::std::denorm_absent>();
    test<long, hip::std::denorm_absent>();
    test<unsigned long, hip::std::denorm_absent>();
    test<long long, hip::std::denorm_absent>();
    test<unsigned long long, hip::std::denorm_absent>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
    test<__int128_t, hip::std::denorm_absent>();
    test<__uint128_t, hip::std::denorm_absent>();
#endif
    test<float, hip::std::denorm_present>();
    test<double, hip::std::denorm_present>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
    test<long double, hip::std::denorm_present>();
#endif

    return 0;
}
