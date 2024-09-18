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

// <cuda/std/tuple>

// constexpr unspecified ignore;

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

__host__ __device__
constexpr bool test_ignore_constexpr()
{
// NVCC does not support constexpr non-integral types
#if TEST_STD_VER > 11 && !defined(__CUDA_ARCH__)
    { // Test that std::ignore provides constexpr converting assignment.
        auto& res = (hip::std::ignore = 42);
        assert(&res == &hip::std::ignore);
    }
    { // Test that hip::std::ignore provides constexpr copy/move constructors
        auto copy = hip::std::ignore;
        auto moved = hip::std::move(copy);
        unused(moved);
    }
    { // Test that hip::std::ignore provides constexpr copy/move assignment
        auto copy = hip::std::ignore;
        copy = hip::std::ignore;
        auto moved = hip::std::ignore;
        moved = hip::std::move(copy);
        unused(moved);
    }
#endif
    return true;
}

int main(int, char**) {
    {
        constexpr auto& ignore_v = hip::std::ignore;
        ((void)ignore_v);
    }
    {
        static_assert(test_ignore_constexpr(), "");
    }
    {
        LIBCPP_STATIC_ASSERT(hip::std::is_trivial<decltype(hip::std::ignore)>::value, "");
    }

  return 0;
}
