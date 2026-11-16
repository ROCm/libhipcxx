//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

template <typename T>
TEST_FUNC void test_impl()
{
  if constexpr (alignof(T) <= 2)
  {
    uintptr_t ptr_int = 10;
    auto ptr          = reinterpret_cast<T>(ptr_int);
    assert(cuda::align_up(ptr, 1) == ptr);
    assert(cuda::align_up(ptr, 2) == ptr);
    assert(cuda::align_up(ptr, 4) == reinterpret_cast<T>(12));
    assert(cuda::align_up(ptr, 8) == reinterpret_cast<T>(16));
  }
  uintptr_t ptr_int2 = 12;
  auto ptr2          = reinterpret_cast<T>(ptr_int2);
  assert(cuda::align_up(ptr2, 8) == reinterpret_cast<T>(16));
}

template <typename T>
TEST_FUNC void test()
{
  test_impl<T*>();
  test_impl<const T*>();
  test_impl<volatile T*>();
  test_impl<const volatile T*>();
}

TEST_FUNC bool test()
{
  test<char>();
  test<short>();
  test<int>();
  test<void>();
  return true;
}

__global__ void test_kernel()
{
  __shared__ int smem_value[4];
  auto ptr = smem_value;
  assert(cuda::device::is_address_from(ptr + 3, cuda::device::address_space::shared));
  assert(cuda::device::is_address_from(cuda::align_up(ptr + 3, 8), cuda::device::address_space::shared));
}

int main(int, char**)
{
  assert(test());
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(); assert(hipDeviceSynchronize() == hipSuccess);))
  return 0;
}
