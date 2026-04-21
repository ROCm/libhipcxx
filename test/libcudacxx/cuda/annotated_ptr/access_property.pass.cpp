//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: nvrtc, hiprtc

#include "utils.h"

template <typename T>
__host__ __device__ __noinline__ void test_global_implicit_property(T ap, cudaAccessProperty cp)
{
  // Test implicit conversions
  cudaAccessProperty v = ap;
  assert(cp == v);

  // Test default, copy constructor, and copy-assignent
  cuda::access_property o(ap);
  cuda::access_property d;
  d = ap;

  // Test explicit conversion to i64
  uint64_t x = (uint64_t) o;
  uint64_t y = (uint64_t) d;
  assert(x == y);
}

__host__ __device__ __noinline__ void test_global()
{
  cuda::access_property o(cuda::access_property::global{});
  uint64_t x = (uint64_t) o;
  unused(x);
}

__host__ __device__ __noinline__ void test_shared()
{
  (void) cuda::access_property::shared{};
}

static_assert(sizeof(cuda::access_property::shared) == 1);
static_assert(sizeof(cuda::access_property::global) == 1);
static_assert(sizeof(cuda::access_property::persisting) == 1);
static_assert(sizeof(cuda::access_property::normal) == 1);
static_assert(sizeof(cuda::access_property::streaming) == 1);
static_assert(sizeof(cuda::access_property) == 8);

static_assert(alignof(cuda::access_property::shared) == 1);
static_assert(alignof(cuda::access_property::global) == 1);
static_assert(alignof(cuda::access_property::persisting) == 1);
static_assert(alignof(cuda::access_property::normal) == 1);
static_assert(alignof(cuda::access_property::streaming) == 1);
static_assert(alignof(cuda::access_property) == 8);

int main(int argc, char** argv)
{
  test_global_implicit_property(cuda::access_property::normal{}, cudaAccessProperty::cudaAccessPropertyNormal);
  test_global_implicit_property(cuda::access_property::streaming{}, cudaAccessProperty::cudaAccessPropertyStreaming);
  test_global_implicit_property(cuda::access_property::persisting{}, cudaAccessProperty::cudaAccessPropertyPersisting);

  test_global();
  test_shared();
  return 0;
}
