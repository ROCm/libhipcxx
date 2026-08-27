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

// UNSUPPORTED: true
// ADDITIONAL_COMPILE_FLAGS: -DUSE_COOPERATIVE_LAUNCH

// <cuda/barrier>
// Verify device-scope completion writes are visible across cooperative blocks.
// Unsupported until the lit wrapper supports cooperative kernel launch.

#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>
#include "test_macros.h"
#include "hip_wavefront_size.h"

static constexpr int k_num_blocks = 2;

__device__ int block0_data;
__device__ int block1_data;

__device__ void cross_block_completion()
{
  block0_data = 100;
  block1_data = 200;
}

using barrier_t = cuda::barrier<cuda::thread_scope_device, void(*)()>;
__device__ alignas(barrier_t) char global_bar_storage[sizeof(barrier_t)];

__device__ int test()
{
  int const k_wave_size = get_wavefront_size();
  int const k_threads_per_block = 2 * k_wave_size;
  int const k_total_threads = k_num_blocks * k_threads_per_block;

  namespace cg = cooperative_groups;
  barrier_t* bar = reinterpret_cast<barrier_t*>(global_bar_storage);

  cg::grid_group grid = cg::this_grid();

  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    init(bar, k_total_threads, cross_block_completion);
    block0_data = 0xBADBAD;
    block1_data = 0xBADBAD;
  }

  // grid.sync() makes initialization visible to all resident blocks.
  grid.sync();

  bar->arrive_and_wait();

  if (blockIdx.x == 0)
  {
    if (block0_data != 100) return 1;
    if (block1_data != 200) return 2;
  }
  else
  {
    if (block0_data != 100) return 3;
    if (block1_data != 200) return 4;
  }

  return 0;
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    int const k_threads_per_block = 2 * k_wave_size;

    cuda_block_count = k_num_blocks;
    cuda_thread_count = k_threads_per_block;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
