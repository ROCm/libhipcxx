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

// UNSUPPORTED: c++03, c++11, c++14

// <memory>

// template <ObjectType T> constexpr T* addressof(T& r);

#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

#if defined(_LIBCUDACXX_ADDRESSOF) || defined(__NVCOMPILER)
struct Pointer {
  __host__ __device__ constexpr Pointer(void* v) : value(v) {}
  void* value;
};

struct A
{
    __host__ __device__ constexpr A() : n(42) {}
    __host__ __device__ void operator&() const { }
    int n; 
};

__device__ constexpr int i = 0;
static_assert(hip::std::addressof(i) == &i, "");

__device__ constexpr double d = 0.0;
static_assert(hip::std::addressof(d) == &d, "");
 
#ifndef __CUDA_ARCH__ // fails in __cudaRegisterVariable
__device__ constexpr A a{};
__device__ constexpr const A* ap = hip::std::addressof(a);
static_assert(&(ap->n) == &(a.n), "");
#endif
#endif

int main(int, char**)
{
  return 0;
}
