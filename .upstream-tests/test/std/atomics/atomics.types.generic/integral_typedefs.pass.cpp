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
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// typedef atomic<char>               atomic_char;
// typedef atomic<signed char>        atomic_schar;
// typedef atomic<unsigned char>      atomic_uchar;
// typedef atomic<short>              atomic_short;
// typedef atomic<unsigned short>     atomic_ushort;
// typedef atomic<int>                atomic_int;
// typedef atomic<unsigned int>       atomic_uint;
// typedef atomic<long>               atomic_long;
// typedef atomic<unsigned long>      atomic_ulong;
// typedef atomic<long long>          atomic_llong;
// typedef atomic<unsigned long long> atomic_ullong;
// typedef atomic<char16_t>           atomic_char16_t;
// typedef atomic<char32_t>           atomic_char32_t;
// typedef atomic<wchar_t>            atomic_wchar_t;
//
// typedef atomic<intptr_t>           atomic_intptr_t;
// typedef atomic<uintptr_t>          atomic_uintptr_t;
//
// typedef atomic<int8_t>             atomic_int8_t;
// typedef atomic<uint8_t>            atomic_uint8_t;
// typedef atomic<int16_t>            atomic_int16_t;
// typedef atomic<uint16_t>           atomic_uint16_t;
// typedef atomic<int32_t>            atomic_int32_t;
// typedef atomic<uint32_t>           atomic_uint32_t;
// typedef atomic<int64_t>            atomic_int64_t;
// typedef atomic<uint64_t>           atomic_uint64_t;

#include <hip/std/atomic>
#include <hip/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((hip::std::is_same<hip::std::atomic<char>, hip::std::atomic_char>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<signed char>, hip::std::atomic_schar>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<unsigned char>, hip::std::atomic_uchar>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<short>, hip::std::atomic_short>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<unsigned short>, hip::std::atomic_ushort>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<int>, hip::std::atomic_int>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<unsigned int>, hip::std::atomic_uint>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<long>, hip::std::atomic_long>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<unsigned long>, hip::std::atomic_ulong>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<long long>, hip::std::atomic_llong>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<unsigned long long>, hip::std::atomic_ullong>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<wchar_t>, hip::std::atomic_wchar_t>::value), "");
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    static_assert((hip::std::is_same<hip::std::atomic<char16_t>, hip::std::atomic_char16_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<char32_t>, hip::std::atomic_char32_t>::value), "");
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS

//  Added by LWG 2441
    static_assert((hip::std::is_same<hip::std::atomic<intptr_t>,  hip::std::atomic_intptr_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<uintptr_t>, hip::std::atomic_uintptr_t>::value), "");

    static_assert((hip::std::is_same<hip::std::atomic<int8_t>,    hip::std::atomic_int8_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<uint8_t>,   hip::std::atomic_uint8_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<int16_t>,   hip::std::atomic_int16_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<uint16_t>,  hip::std::atomic_uint16_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<int32_t>,   hip::std::atomic_int32_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<uint32_t>,  hip::std::atomic_uint32_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<int64_t>,   hip::std::atomic_int64_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<uint64_t>,  hip::std::atomic_uint64_t>::value), "");

  return 0;
}
