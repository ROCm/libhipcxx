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

#ifndef SUPPORT_TEST_CONVERTIBLE_H
#define SUPPORT_TEST_CONVERTIBLE_H

// "test_convertible<Tp, Args...>()" is a metafunction used to check if 'Tp'
// is implicitly convertible from 'Args...' for any number of arguments,
// Unlike 'std::is_convertible' which only allows checking for single argument
// conversions.

#include <hip/std/type_traits>

#include "test_macros.h"

#if TEST_STD_VER < 11
#error test_convertible.h requires C++11 or newer
#endif

namespace detail {
    template <class Tp> __host__ __device__ void eat_type(Tp);

    template <class Tp, class ...Args>
    __host__ __device__ constexpr auto test_convertible_imp(int)
        -> decltype(eat_type<Tp>({hip::std::declval<Args>()...}), true)
    { return true; }

    template <class Tp, class ...Args>
    __host__ __device__ constexpr auto test_convertible_imp(long) -> bool { return false; }
}

template <class Tp, class ...Args>
__host__ __device__ constexpr bool test_convertible()
{ return detail::test_convertible_imp<Tp, Args...>(0); }

#endif // SUPPORT_TEST_CONVERTIBLE_H
