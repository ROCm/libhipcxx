//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// UNSUPPORTED: msvc, windows_amd

// cuda::std::ranges::end
// cuda::std::ranges::cend
//   Test the fix for https://llvm.org/PR54100

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

#ifndef __CUDA_ARCH__
struct A
{
  int m[0];
};
static_assert(sizeof(A) == 0); // an extension supported by GCC and Clang

__device__ static A a[10];

int main(int, char**)
{
  auto p = cuda::std::ranges::end(a);
  static_assert(cuda::std::same_as<A*, decltype(cuda::std::ranges::end(a))>);
  assert(p == a + 10);
  auto cp = cuda::std::ranges::cend(a);
  static_assert(cuda::std::same_as<const A*, decltype(cuda::std::ranges::cend(a))>);
  assert(cp == a + 10);

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
