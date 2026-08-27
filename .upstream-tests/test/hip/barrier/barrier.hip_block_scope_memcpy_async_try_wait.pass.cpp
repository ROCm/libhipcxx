//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// UNSUPPORTED: pre-gfx1250
// UNSUPPORTED: nvcc, nvhpc, nvc++

// <cuda/barrier>

// Verify public try-wait endpoints observe pending tx work registered by
// ordinary cuda::memcpy_async(..., barrier) on the HIP LDS-backed block barrier
// path. The test observes the pending state from the copy's group object before
// the copy dispatch has a chance to complete the registered tx work.

#include <cuda/barrier>
#include <cuda/std/chrono>
#include <cuda/std/utility>
#include "test_macros.h"

constexpr int k_count = 4096;

__device__ __attribute__((aligned(16))) int g_src[k_count];

struct probing_group
{
  cuda::barrier<cuda::thread_scope_block>* bar;
  cuda::barrier<cuda::thread_scope_block>::arrival_token* token;
  int* result;
  bool* probed;

  _CCCL_DEVICE void sync() const {}

  _CCCL_DEVICE cuda::std::size_t size() const
  {
    probe();
    return 1;
  }

  _CCCL_DEVICE cuda::std::size_t thread_rank() const
  {
    return 0;
  }

  _CCCL_DEVICE void probe() const
  {
    if (*probed || *result != 0) {
      return;
    }

    *probed = true;
    *token = bar->arrive(1);
    auto past = cuda::std::chrono::high_resolution_clock::now() - cuda::std::chrono::seconds(1);

    auto pending_for_token = *token;
    bool const completed_for = bar->try_wait_for(
      cuda::std::move(pending_for_token),
      cuda::std::chrono::nanoseconds(0));
    if (completed_for) {
      *result = 1;
      return;
    }

    auto pending_until_token = *token;
    bool const completed_until = bar->try_wait_until(
      cuda::std::move(pending_until_token),
      past);
    if (completed_until) {
      *result = 2;
      return;
    }

    bool const completed_parity_for = bar->try_wait_parity_for(
      false,
      cuda::std::chrono::nanoseconds(0));
    if (completed_parity_for) {
      *result = 3;
      return;
    }

    bool const completed_parity_until = bar->try_wait_parity_until(false, past);
    if (completed_parity_until) {
      *result = 4;
    }
  }
};

__device__ int test()
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  __shared__ __attribute__((aligned(16))) int dest[k_count];
  __shared__ cuda::barrier<cuda::thread_scope_block>::arrival_token token;
  __shared__ int result;
  __shared__ bool probed;

  if (threadIdx.x == 0) {
    init(&bar, 1);
    result = 0;
    probed = false;
    for (int i = 0; i < k_count; ++i) {
      g_src[i] = 1000 + i;
      dest[i] = -1;
    }

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

    probing_group group{&bar, &token, &result, &probed};
    cuda::memcpy_async(
      group,
      &dest[0],
      &g_src[0],
      cuda::aligned_size_t<16>(sizeof(int) * k_count),
      bar);

    if (!probed && result == 0) {
      return 10;
    }
    if (result != 0) {
      return result;
    }

    auto past = cuda::std::chrono::high_resolution_clock::now() - cuda::std::chrono::seconds(1);

    auto wait_token = token;
    bar.wait(cuda::std::move(wait_token));

    for (int i = 0; i < k_count; ++i) {
      if (dest[i] != 1000 + i) {
        return 5;
      }
    }

    auto completed_for_token = token;
    bool const completed_for_after_copy = bar.try_wait_for(
      cuda::std::move(completed_for_token),
      cuda::std::chrono::nanoseconds(0));
    if (!completed_for_after_copy) {
      return 6;
    }

    auto completed_until_token = token;
    bool const completed_until_after_copy = bar.try_wait_until(
      cuda::std::move(completed_until_token),
      past);
    if (!completed_until_after_copy) {
      return 7;
    }

    bool const completed_parity_for_after_copy = bar.try_wait_parity_for(
      false,
      cuda::std::chrono::nanoseconds(0));
    if (!completed_parity_for_after_copy) {
      return 8;
    }

    bool const completed_parity_until_after_copy = bar.try_wait_parity_until(false, past);
    if (!completed_parity_until_after_copy) {
      return 9;
    }
  }

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count = 1;
    cuda_thread_count = 1;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (return test();))
  return 0;
}