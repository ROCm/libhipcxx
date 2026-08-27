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

#include <cuda/barrier>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "hip_wavefront_size.h"

// Test barrier_native_handle for AMD block-scope barriers.
//
// hip::device::barrier_native_handle(barrier&) returns a raw pointer to the
// barrier's active native phase state.
//
// Contract:
//   - Pointer is non-null
//   - Pointer address is within barrier storage bounds [&b, &b + sizeof(b))
//   - Barrier remains functional after handle access
//
// Use cases:
//   1. Testing/debugging (inspect packed word directly)
//   2. Future inline assembly (if needed)
//   3. API compatibility with CUDA's barrier_native_handle()

__device__ hip::barrier<hip::thread_scope_block> g_device_barrier;

__device__ bool verify_software_fallback_state(hip::std::uint64_t* handle, int expected_count)
{
  hip::std::uint64_t packed_word = *handle;
  hip::std::uint32_t arrived = static_cast<hip::std::uint32_t>(packed_word >> 32);
  hip::std::uint32_t expected = static_cast<hip::std::uint32_t>(packed_word & 0xFFFFFFFF);

  hip::std::uint32_t init_value = (1u << 31) - expected_count;
  return arrived == init_value && expected == init_value;
}

__device__ int test_shared()
{
  __shared__ hip::barrier<hip::thread_scope_block> barrier;

  if (threadIdx.x == 0)
  {
    init(&barrier, blockDim.x);
  }
  __syncthreads();

  auto* handle = hip::device::barrier_native_handle(barrier);

  if (handle == nullptr) return 1;

  auto* barrier_start = reinterpret_cast<hip::std::uint64_t*>(&barrier);
  auto* barrier_end   = reinterpret_cast<hip::std::uint64_t*>(&barrier + 1);

  if (handle < barrier_start) return 1;
  if (handle >= barrier_end) return 1;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1250__)
  // Shared barriers use the LDS phase object on gfx1250.
  hip::std::uint64_t packed_word = *handle;
  hip::std::uint32_t pending_count = static_cast<hip::std::uint32_t>(packed_word & hip::__lds_barrier_t::pending_count_max);
  hip::std::uint32_t init_count = static_cast<hip::std::uint32_t>((packed_word >> 32) & hip::__lds_barrier_t::init_count_max);

  hip::std::uint32_t init_value = blockDim.x - 1;
  if (pending_count != init_value) return 1;
  if (init_count != init_value) return 1;
#else
  if (!verify_software_fallback_state(handle, blockDim.x)) return 1;
#endif

  barrier.arrive_and_wait();

  return 0;
}

__device__ int test_device()
{
  if (threadIdx.x == 0)
  {
    init(&g_device_barrier, blockDim.x);
  }
  __syncthreads();

  auto* handle = hip::device::barrier_native_handle(g_device_barrier);

  if (handle == nullptr) return 1;

  auto* barrier_start = reinterpret_cast<hip::std::uint64_t*>(&g_device_barrier);
  auto* barrier_end   = reinterpret_cast<hip::std::uint64_t*>(&g_device_barrier + 1);

  if (handle < barrier_start) return 1;
  if (handle >= barrier_end) return 1;

  // Device-global barriers use the software fallback state.
  if (!verify_software_fallback_state(handle, blockDim.x)) return 1;

  g_device_barrier.arrive_and_wait();

  return 0;
}

__device__ int test()
{
  int result = 0;

  result = test_shared();
  if (result != 0) return result;

  result = test_device();
  return result;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_thread_count = get_wavefront_size();

    cuda_block_count = 1;
    cuda_thread_count = k_thread_count;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (return test();))

  return 0;
}
