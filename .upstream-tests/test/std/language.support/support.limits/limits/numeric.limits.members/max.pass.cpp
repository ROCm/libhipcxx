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

// max()

#include <hip/std/limits>
#include <hip/std/climits>
#include <hip/std/cfloat>
#include <hip/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__
void
test(T expected)
{
    assert(hip::std::numeric_limits<T>::max() == expected);
    assert(hip::std::numeric_limits<T>::is_bounded);
    assert(hip::std::numeric_limits<const T>::max() == expected);
    assert(hip::std::numeric_limits<const T>::is_bounded);
    assert(hip::std::numeric_limits<volatile T>::max() == expected);
    assert(hip::std::numeric_limits<volatile T>::is_bounded);
    assert(hip::std::numeric_limits<const volatile T>::max() == expected);
    assert(hip::std::numeric_limits<const volatile T>::is_bounded);
}

int main(int, char**)
{
#ifndef _LIBCUDACXX_COMPILER_NVRTC
    test<wchar_t>(WCHAR_MAX);
#endif
    test<bool>(true);
    test<char>(CHAR_MAX);
    test<signed char>(SCHAR_MAX);
    test<unsigned char>(UCHAR_MAX);
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>(UCHAR_MAX); // ??
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<char16_t>(USHRT_MAX);
    test<char32_t>(UINT_MAX);
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    test<short>(SHRT_MAX);
    test<unsigned short>(USHRT_MAX);
    test<int>(INT_MAX);
    test<unsigned int>(UINT_MAX);
    test<long>(LONG_MAX);
    test<unsigned long>(ULONG_MAX);
    test<long long>(LLONG_MAX);
    test<unsigned long long>(ULLONG_MAX);
#ifndef _LIBCUDACXX_HAS_NO_INT128
    test<__int128_t>(__int128_t(__uint128_t(-1)/2));
    test<__uint128_t>(__uint128_t(-1));
#endif
    test<float>(FLT_MAX);
    test<double>(DBL_MAX);
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
    test<long double>(LDBL_MAX);
#endif

    return 0;
}
