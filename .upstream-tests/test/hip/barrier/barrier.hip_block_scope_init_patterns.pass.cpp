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

// UNSUPPORTED: nvcc, nvhpc, nvc++

// <cuda/barrier>
// Verify construction, zero-count init, and safe reinitialization patterns.
// Reinit occurs only after the prior phase is quiesced.

#include <cuda/barrier>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include "test_macros.h"
#include "hip_wavefront_size.h"

// hip::barrier<thread_scope_block> layout on gfx1250 device builds:
//   [0] __barrier_base (8 bytes) - fallback software phase state
//   [8] __tx_barrier   (8 bytes) - active LDS phase object for shared barriers
// Host builds and targets without an LDS phase object keep only the fallback
// software phase state.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1250__)
static_assert(sizeof(hip::barrier<hip::thread_scope_block>) == 2 * sizeof(hip::std::uint64_t),
              "hip::barrier<thread_scope_block> layout assumption violated");
#else
static_assert(sizeof(hip::barrier<hip::thread_scope_block>) == sizeof(hip::std::uint64_t),
              "hip::barrier<thread_scope_block> layout assumption violated");
#endif

// Decode the expected arrival count from the active native state word.
// On gfx1250 shared barriers, the active LDS phase object stores
// expected-1 in init_count. Fallback software state encodes expected in the
// lower 32 bits as (2^31 - expected_count).
// Call only immediately after init() and before any arrive().
__device__ static hip::std::int32_t
decode_expected(hip::barrier<hip::thread_scope_block>& b)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1250__)
  hip::std::uint64_t raw = *hip::device::barrier_native_handle(b);
  hip::std::uint32_t init_count = static_cast<hip::std::uint32_t>((raw >> 32) & 0xFFFFu);
  return static_cast<hip::std::int32_t>(init_count + 1);
#else
// The expected field (bits 0-31) is encoded as (2^31 - expected_count), so
// expected_count = (2^31 - lower_32_bits).
  static constexpr hip::std::uint64_t k_expected_mask = (1ull << 32) - 1;
  hip::std::uint64_t raw = *hip::device::barrier_native_handle(b);
  return static_cast<hip::std::int32_t>(
      (1u << 31) - static_cast<hip::std::uint32_t>(raw & k_expected_mask));
#endif
}

__device__ int test_constructor_via_init()
{
  int const k_wave_size = get_wavefront_size();
  int const k_thread_count = 2 * k_wave_size;

  __shared__ hip::barrier<hip::thread_scope_block> b;
  if (threadIdx.x == 0)
    init(&b, k_thread_count);
  __syncthreads();

  b.arrive_and_wait();
  return 0;
}

__device__ int test_reinit_two_waves()
{
  int const k_wave_size = get_wavefront_size();
  int const k_thread_count = 2 * k_wave_size;

  __shared__ hip::barrier<hip::thread_scope_block> b;
  __shared__ int result;

  if (threadIdx.x == 0)
    result = 0;
  if (threadIdx.x == 0)
    init(&b, k_thread_count);
  __syncthreads();

  if (threadIdx.x == 0)
  {
    if (decode_expected(b) != k_thread_count)
      result = 1;
  }

  b.arrive_and_wait();

  // Complete the current phase before reinitializing the object.
  b.arrive_and_wait();
  __syncthreads();

  if (threadIdx.x == 0)
    init(&b, k_wave_size);
  __syncthreads();

  if (threadIdx.x == 0)
  {
    if (result == 0 && decode_expected(b) != k_wave_size)
      result = 2;
  }

  // Only wave 0 participates after reinit with expected == k_wave_size.
  if (threadIdx.x < k_wave_size)
    b.arrive_and_wait();
  __syncthreads();

  return result;
}

__device__ int test()
{
  int result = 0;

  result = test_constructor_via_init();
  if (result != 0) return result;

  result = test_reinit_two_waves();
  return result;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    int const k_thread_count = 2 * k_wave_size;

    cuda_block_count  = 1;
    cuda_thread_count = k_thread_count;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (return test();))
  return 0;
}
