//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: gcc-8, gcc-9
// UNSUPPORTED: windows && (c++11 || c++14 || c++17)

// template<class F, class... Args>
// concept predicate;

#include <hip/std/concepts>

#include "test_macros.h"
#if TEST_STD_VER > 17

__host__ __device__ constexpr bool check_subsumption(hip::std::regular_invocable auto) {
  return false;
}

// clang-format off
template<class F>
requires hip::std::predicate<F> && true
__host__ __device__ constexpr bool check_subsumption(F)
{
  return true;
}
// clang-format on
struct not_predicate {
  __host__ __device__ constexpr void operator()() const {}
};

struct predicate {
  __host__ __device__ constexpr bool operator()() const { return true; }
};

static_assert(!check_subsumption(not_predicate{}), "");
static_assert(check_subsumption(predicate{}), "");

#endif // TEST_STD_VER > 17

int main(int, char**)
{
  return 0;
}
