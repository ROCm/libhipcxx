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

// signaling_NaN()

#include <hip/std/limits>
#include <hip/std/cmath>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__
void
test_imp(hip::std::true_type)
{
    assert(hip::std::isnan(hip::std::numeric_limits<T>::signaling_NaN()));
    assert(hip::std::isnan(hip::std::numeric_limits<const T>::signaling_NaN()));
    assert(hip::std::isnan(hip::std::numeric_limits<volatile T>::signaling_NaN()));
    assert(hip::std::isnan(hip::std::numeric_limits<const volatile T>::signaling_NaN()));
}

template <class T>
__host__ __device__
void
test_imp(hip::std::false_type)
{
    assert(hip::std::numeric_limits<T>::signaling_NaN() == T());
    assert(hip::std::numeric_limits<const T>::signaling_NaN() == T());
    assert(hip::std::numeric_limits<volatile T>::signaling_NaN() == T());
    assert(hip::std::numeric_limits<const volatile T>::signaling_NaN() == T());
}

template <class T>
__host__ __device__
inline
void
test()
{
    test_imp<T>(hip::std::is_floating_point<T>());
}

int main(int, char**)
{
    test<bool>();
    test<char>();
    test<signed char>();
    test<unsigned char>();
    test<wchar_t>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<char16_t>();
    test<char32_t>();
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<short>();
    test<unsigned short>();
    test<int>();
    test<unsigned int>();
    test<long>();
    test<unsigned long>();
    test<long long>();
    test<unsigned long long>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
    test<__int128_t>();
    test<__uint128_t>();
#endif
    test<float>();
    test<double>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
    test<long double>();
#endif

    return 0;
}
