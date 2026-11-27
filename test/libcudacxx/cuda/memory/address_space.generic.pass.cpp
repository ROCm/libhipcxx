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

// UNSUPPORTED: nvrtc, hiprtc, enable-tile
// Optimization can hide the SPIR-V address-space cast mismatch.
// ADDITIONAL_COMPILE_OPTIONS_HOST: -O0

#include <cuda/memory>

#include "test_macros.h"

// Check both matching and non-matching spaces through a generic pointer.
__device__ __noinline__ int classify_address(const volatile int* ptr)
{
  using cuda::device::address_space;
  using cuda::device::is_address_from;
  return (is_address_from(ptr, address_space::local) ? 1 : 0) | (is_address_from(ptr, address_space::shared) ? 2 : 0)
       | (is_address_from(ptr, address_space::global) ? 4 : 0);
}

__global__ void test_kernel(int* results)
{
  __shared__ int shared_var;
  int local_var{};
  results[0] = classify_address(results);
  results[1] = classify_address(&shared_var);
  results[2] = classify_address(&local_var);
}

void test()
{
  int* device_results = nullptr;
  int results[3]{};
  assert(hipMalloc(&device_results, sizeof(results)) == hipSuccess);
  test_kernel<<<1, 1>>>(device_results);
  assert(hipGetLastError() == hipSuccess);
  assert(hipMemcpy(results, device_results, sizeof(results), hipMemcpyDeviceToHost) == hipSuccess);
  assert(hipFree(device_results) == hipSuccess);
  assert(results[0] == 4);
  assert(results[1] == 2);
  assert(results[2] == 1);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
