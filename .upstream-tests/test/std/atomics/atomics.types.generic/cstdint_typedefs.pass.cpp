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

// typedef atomic<int_least8_t>   atomic_int_least8_t;
// typedef atomic<uint_least8_t>  atomic_uint_least8_t;
// typedef atomic<int_least16_t>  atomic_int_least16_t;
// typedef atomic<uint_least16_t> atomic_uint_least16_t;
// typedef atomic<int_least32_t>  atomic_int_least32_t;
// typedef atomic<uint_least32_t> atomic_uint_least32_t;
// typedef atomic<int_least64_t>  atomic_int_least64_t;
// typedef atomic<uint_least64_t> atomic_uint_least64_t;
//
// typedef atomic<int_fast8_t>   atomic_int_fast8_t;
// typedef atomic<uint_fast8_t>  atomic_uint_fast8_t;
// typedef atomic<int_fast16_t>  atomic_int_fast16_t;
// typedef atomic<uint_fast16_t> atomic_uint_fast16_t;
// typedef atomic<int_fast32_t>  atomic_int_fast32_t;:q:q!
// typedef atomic<uint_fast32_t> atomic_uint_fast32_t;
// typedef atomic<int_fast64_t>  atomic_int_fast64_t;
// typedef atomic<uint_fast64_t> atomic_uint_fast64_t;
//
// typedef atomic<intptr_t>  atomic_intptr_t;
// typedef atomic<uintptr_t> atomic_uintptr_t;
// typedef atomic<size_t>    atomic_size_t;
// typedef atomic<ptrdiff_t> atomic_ptrdiff_t;
// typedef atomic<intmax_t>  atomic_intmax_t;
// typedef atomic<uintmax_t> atomic_uintmax_t;

#include <hip/std/atomic>
#include <hip/std/type_traits>
#include <hip/std/cstdint>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((hip::std::is_same<hip::std::atomic<  hip::std::int_least8_t>,   hip::std::atomic_int_least8_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic< hip::std::uint_least8_t>,  hip::std::atomic_uint_least8_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic< hip::std::int_least16_t>,  hip::std::atomic_int_least16_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<hip::std::uint_least16_t>, hip::std::atomic_uint_least16_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic< hip::std::int_least32_t>,  hip::std::atomic_int_least32_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<hip::std::uint_least32_t>, hip::std::atomic_uint_least32_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic< hip::std::int_least64_t>,  hip::std::atomic_int_least64_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<hip::std::uint_least64_t>, hip::std::atomic_uint_least64_t>::value), "");

    static_assert((hip::std::is_same<hip::std::atomic<  hip::std::int_fast8_t>,   hip::std::atomic_int_fast8_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic< hip::std::uint_fast8_t>,  hip::std::atomic_uint_fast8_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic< hip::std::int_fast16_t>,  hip::std::atomic_int_fast16_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<hip::std::uint_fast16_t>, hip::std::atomic_uint_fast16_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic< hip::std::int_fast32_t>,  hip::std::atomic_int_fast32_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<hip::std::uint_fast32_t>, hip::std::atomic_uint_fast32_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic< hip::std::int_fast64_t>,  hip::std::atomic_int_fast64_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<hip::std::uint_fast64_t>, hip::std::atomic_uint_fast64_t>::value), "");

    static_assert((hip::std::is_same<hip::std::atomic< hip::std::intptr_t>,  hip::std::atomic_intptr_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<hip::std::uintptr_t>, hip::std::atomic_uintptr_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<   hip::std::size_t>,    hip::std::atomic_size_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<hip::std::ptrdiff_t>, hip::std::atomic_ptrdiff_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic< hip::std::intmax_t>,  hip::std::atomic_intmax_t>::value), "");
    static_assert((hip::std::is_same<hip::std::atomic<hip::std::uintmax_t>, hip::std::atomic_uintmax_t>::value), "");

  return 0;
}
