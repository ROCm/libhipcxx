// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: nvcc, nvhpc, nvc++

// <hip/barrier>

// Verify ordinary hip::memcpy_async falls back to a real synchronous copy when
// the copy direction is not accelerated by the HIP async LDS path.

#include <hip/barrier>
#include <hip/std/utility>
#include "test_macros.h"

constexpr int k_count = 16;

__device__ __attribute__((aligned(16))) int g_src[k_count] = {
  1100, 1101, 1102, 1103,
  1104, 1105, 1106, 1107,
  1108, 1109, 1110, 1111,
  1112, 1113, 1114, 1115
};

__device__ __attribute__((aligned(16))) int g_dest[k_count];

__device__ int test()
{
  __shared__ hip::barrier<hip::thread_scope_block> bar;

  if (threadIdx.x == 0) {
    init(&bar, 1);
    for (int index = 0; index < k_count; ++index) {
      g_dest[index] = -1;
    }

    hip::memcpy_async(
      &g_dest[0],
      &g_src[0],
      hip::aligned_size_t<16>(sizeof(int) * k_count),
      bar);

    auto token = bar.arrive();
    bar.wait(hip::std::move(token));

    for (int index = 0; index < k_count; ++index) {
      if (g_dest[index] != g_src[index]) {
        return 1;
      }
    }
  }

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count  = 1;
    cuda_thread_count = 1;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (return test();))
  return 0;
}
