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

// UNSUPPORTED: pre-gfx1250
// UNSUPPORTED: nvcc, nvhpc, nvc++

// <hip/barrier>

// Verify four-phase reuse with ordinary public hip::memcpy_async producers.

#include <hip/barrier>
#include <hip/std/chrono>
#include <hip/std/utility>
#include "test_macros.h"

constexpr int k_count = 16;
constexpr int k_total = 4 * k_count;

__device__ __attribute__((aligned(16))) int g_src[k_total] = {
  500, 501, 502, 503, 504, 505, 506, 507,
  508, 509, 510, 511, 512, 513, 514, 515,
  600, 601, 602, 603, 604, 605, 606, 607,
  608, 609, 610, 611, 612, 613, 614, 615,
  700, 701, 702, 703, 704, 705, 706, 707,
  708, 709, 710, 711, 712, 713, 714, 715,
  800, 801, 802, 803, 804, 805, 806, 807,
  808, 809, 810, 811, 812, 813, 814, 815
};

__device__ int validate_range(int const* dest, int offset)
{
  for (int index = 0; index < k_count; ++index) {
    int const absolute = offset + index;
    if (dest[absolute] != g_src[absolute]) {
      return 1;
    }
  }
  return 0;
}

__device__ int test()
{
  __shared__ hip::barrier<hip::thread_scope_block> bar;
  __shared__ __attribute__((aligned(16))) int dest[k_total];

  if (threadIdx.x == 0) {
    init(&bar, 1);
    for (int index = 0; index < k_total; ++index) {
      dest[index] = -1;
    }

    hip::memcpy_async(
      &dest[0],
      &g_src[0],
      hip::aligned_size_t<16>(sizeof(int) * k_count),
      bar);
    auto phase0_token = bar.arrive();
    bar.wait(hip::std::move(phase0_token));
    if (validate_range(dest, 0) != 0) {
      return 1;
    }

    auto phase1_token = bar.arrive();
    if (!bar.try_wait_for(hip::std::move(phase1_token), hip::std::chrono::nanoseconds(0))) {
      return 2;
    }

    hip::memcpy_async(
      &dest[k_count],
      &g_src[k_count],
      hip::aligned_size_t<16>(sizeof(int) * k_count),
      bar);
    hip::memcpy_async(
      &dest[2 * k_count],
      &g_src[2 * k_count],
      hip::aligned_size_t<16>(sizeof(int) * k_count),
      bar);
    auto phase2_token = bar.arrive();
    bar.wait(hip::std::move(phase2_token));
    if (validate_range(dest, k_count) != 0) {
      return 3;
    }
    if (validate_range(dest, 2 * k_count) != 0) {
      return 4;
    }

    hip::memcpy_async(
      &dest[3 * k_count],
      &g_src[3 * k_count],
      hip::aligned_size_t<16>(sizeof(int) * k_count),
      bar);
    auto phase3_token = bar.arrive();
    bar.wait(hip::std::move(phase3_token));
    if (validate_range(dest, 3 * k_count) != 0) {
      return 5;
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
