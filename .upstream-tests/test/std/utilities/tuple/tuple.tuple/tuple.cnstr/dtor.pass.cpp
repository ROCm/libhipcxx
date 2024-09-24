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

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

// <cuda/std/tuple>

// template <class... Types> class tuple;

// ~tuple();

// C++17 added:
//   The destructor of tuple shall be a trivial destructor
//     if (is_trivially_destructible_v<Types> && ...) is true.

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"

struct not_trivially_destructible {
    __host__ __device__ virtual ~not_trivially_destructible() {}
};

int main(int, char**)
{
    static_assert(hip::std::is_trivially_destructible<
        hip::std::tuple<> >::value, "");
    static_assert(hip::std::is_trivially_destructible<
        hip::std::tuple<void*> >::value, "");
    static_assert(hip::std::is_trivially_destructible<
        hip::std::tuple<int, float> >::value, "");
    // hip::std::string is not supported
    /*
    static_assert(!hip::std::is_trivially_destructible<
        hip::std::tuple<not_trivially_destructible> >::value, "");
    static_assert(!hip::std::is_trivially_destructible<
        hip::std::tuple<int, not_trivially_destructible> >::value, "");
    */
    // non-string check
    static_assert(!hip::std::is_trivially_destructible<
        hip::std::tuple<not_trivially_destructible> >::value, "");
    static_assert(!hip::std::is_trivially_destructible<
        hip::std::tuple<int, not_trivially_destructible> >::value, "");
  return 0;
}
