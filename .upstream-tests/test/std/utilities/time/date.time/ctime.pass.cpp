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

#include <hip/std/ctime>
#include <hip/std/type_traits>

#include "test_macros.h"

#ifndef NULL
#error NULL not defined
#endif

#ifndef __CUDACC_RTC__
#ifndef CLOCKS_PER_SEC
#error CLOCKS_PER_SEC not defined
#endif
#endif

#if TEST_STD_VER > 14 && defined(TEST_HAS_C11_FEATURES)
#ifndef TIME_UTC
#error TIME_UTC not defined
#endif
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wformat-zero-length"
#endif

#pragma nv_diag_suppress set_but_not_used

int main(int, char**)
{
    hip::std::clock_t c = 0;
    hip::std::size_t s = 0;
    hip::std::time_t t = 0;
    ((void)c); // Prevent unused warning
    ((void)s); // Prevent unused warning
    ((void)t); // Prevent unused warning
#ifndef __CUDACC_RTC__
    hip::std::tm tm = {};
    char str[3];
    ((void)tm); // Prevent unused warning
    ((void)str); // Prevent unused warning
#if TEST_STD_VER > 14 && defined(TEST_HAS_C11_FEATURES)
    hip::std::timespec tmspec = {};
    ((void)tmspec); // Prevent unused warning
#endif

    //FIXME(hip): clock() is declared as extern clock_t clock (void) __THROW; in <time.h>. clock_t is a typedef to long.
    //The below test doesn't work for HIP, as the clock() function is defined in /opt/rocm/include/hip/amd_detail/amd_device_functions.h
    //as "long long int  clock() { return __clock(); }". Therefore, the decltype expands to long long instead of the expected type long.
#ifndef __HIP__
    static_assert((hip::std::is_same<decltype(hip::std::clock()), hip::std::clock_t>::value), "");
#endif
    static_assert((hip::std::is_same<decltype(hip::std::difftime(t,t)), double>::value), "");
    static_assert((hip::std::is_same<decltype(hip::std::mktime(&tm)), hip::std::time_t>::value), "");
    static_assert((hip::std::is_same<decltype(hip::std::time(&t)), hip::std::time_t>::value), "");
#if TEST_STD_VER > 14 && defined(TEST_HAS_TIMESPEC_GET)
    static_assert((hip::std::is_same<decltype(hip::std::timespec_get(&tmspec, 0)), int>::value), "");
#endif
#ifndef _LIBCUDACXX_HAS_NO_THREAD_UNSAFE_C_FUNCTIONS
    static_assert((hip::std::is_same<decltype(hip::std::asctime(&tm)), char*>::value), "");
    static_assert((hip::std::is_same<decltype(hip::std::ctime(&t)), char*>::value), "");
    static_assert((hip::std::is_same<decltype(hip::std::gmtime(&t)), hip::std::tm*>::value), "");
    static_assert((hip::std::is_same<decltype(hip::std::localtime(&t)), hip::std::tm*>::value), "");
#endif
    static_assert((hip::std::is_same<decltype(hip::std::strftime(str,s,"",&tm)), hip::std::size_t>::value), "");
#endif

  return 0;
}
