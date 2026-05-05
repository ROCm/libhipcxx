//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
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

// UNSUPPORTED: pre-sm-70, hipcc, hiprtc
// NOTE(HIP/AMD): warp_match_all/warp_match_any rely on the match.{any,all}.sync
// PTX instructions which have no equivalent on AMD/HIP hardware.

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

#include "test_macros.h"

template <typename T>
__device__ void test_types(T valueA = T{}, T valueB = T{1})
{
  for (int i = 1; i < 32; ++i)
  {
    auto mask = cuda::device::lane_mask{(1u << i) - 1};
    assert(cuda::device::warp_match_all(valueA, mask));
    if (i > 1)
    {
      [[maybe_unused]] auto value = threadIdx.x == 0 ? valueA : valueB;
      assert(!cuda::device::warp_match_all(value, mask));
    }
  }
}

__global__ void test_kernel()
{
  test_types<uint8_t>();
  test_types<uint16_t>();
  test_types<uint32_t>();
  test_types<uint64_t>();
#if _CCCL_HAS_INT128()
  test_types<__uint128_t>();
#endif
  test_types(char3{0, 0, 0}, char3{1, 1, 1});
  using array_t = cuda::std::array<char, 6>;
  test_types(array_t{0, 0, 0, 0, 0, 0}, array_t{1, 1, 1, 1, 1, 1});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 32>>>();))
  return 0;
}
