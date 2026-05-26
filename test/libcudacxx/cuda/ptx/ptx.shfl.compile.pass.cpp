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
// Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: clang && !nvcc

// <cuda/ptx>

// NOTE(HIP/AMD): on HIP the public umbrella <cuda/ptx> hard-errors
// (cuda::ptx is an NV-only public API per the design); route through
// the individual instruction header instead, which carries the HIP
// software-emulated implementation behind a _CCCL_HIP_COMPILATION()
// gate.
#ifdef __HIP_PLATFORM_AMD__
#  include <cuda/__ptx/instructions/shfl_sync.h>
#else
#  include <cuda/ptx>
#endif
#include <cuda/std/utility>

// NOTE(HIP/AMD): the upstream test_shfl_* bodies below are gated on
// '__cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()'. The first half
// is false on HIP (PTX-ISA is 0), so the existing assertion blocks
// compile out cleanly. Force HIP overload instantiation explicitly via
// a function-pointer block so this compile.pass.cpp exercises the HIP
// wrappers in <cuda/__ptx/instructions/shfl_sync.h>. Documented
// HIP-arm divergences (per the file-header NOTE there): on wave-64 the
// whole-warp (segmask=0x1f) shuffle width is warpSize=64 instead of 32,
// and the 'pred' out-param is always true (HIP __shfl* implicitly syncs
// across the active wave -- no per-lane membermask membership info
// exposed). Runtime parity vs CUDA PTX is tracked as a follow-up
// (g32 thread, PR #217).
#if _CCCL_HIP_COMPILATION()
__global__ void test_shfl_sync_hip(void** __fn_ptr)
{
  // Pick uint32_t as the representative 4-byte type ('shfl.sync.b32'
  // is the only width PTX supports).
  using __u32 = cuda::std::uint32_t;
  // shfl_sync_idx: with-pred and without-pred overloads
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<__u32 (*)(__u32, bool&, __u32, __u32, __u32)>(cuda::ptx::shfl_sync_idx));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<__u32 (*)(__u32, __u32, __u32, __u32)>(cuda::ptx::shfl_sync_idx));
  // shfl_sync_up
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<__u32 (*)(__u32, bool&, __u32, __u32, __u32)>(cuda::ptx::shfl_sync_up));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<__u32 (*)(__u32, __u32, __u32, __u32)>(cuda::ptx::shfl_sync_up));
  // shfl_sync_down
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<__u32 (*)(__u32, bool&, __u32, __u32, __u32)>(cuda::ptx::shfl_sync_down));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<__u32 (*)(__u32, __u32, __u32, __u32)>(cuda::ptx::shfl_sync_down));
  // shfl_sync_bfly
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<__u32 (*)(__u32, bool&, __u32, __u32, __u32)>(cuda::ptx::shfl_sync_bfly));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<__u32 (*)(__u32, __u32, __u32, __u32)>(cuda::ptx::shfl_sync_bfly));
}
#endif // _CCCL_HIP_COMPILATION()

__host__ __device__ void test_shfl_full_mask()
{
#if __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
  constexpr unsigned FullMask = 0xFFFFFFFF;
  auto data                   = threadIdx.x;
  bool pred1, pred2, pred3, pred4;
  auto res1 = cuda::ptx::shfl_sync_idx(data, pred1, 2 /*idx*/, 0b11111 /*clamp*/, FullMask);
  assert(res1 == 2 && pred1);

  auto res2 = cuda::ptx::shfl_sync_up(data, pred2, 2 /*offset*/, 0 /*clamp*/, FullMask);
  if (threadIdx.x <= 1)
  {
    assert(res2 == threadIdx.x && !pred2);
  }
  else
  {
    assert(res2 == threadIdx.x - 2 && pred2);
  }

  auto res3 = cuda::ptx::shfl_sync_down(data, pred3, 2 /*offset*/, 0b11111 /*clamp*/, FullMask);
  if (threadIdx.x >= 30)
  {
    assert(res3 == threadIdx.x && !pred3);
  }
  else
  {
    assert(res3 == threadIdx.x + 2 && pred3);
  }

  auto res4 = cuda::ptx::shfl_sync_bfly(data, pred4, 2 /*offset*/, 0b11111 /*clamp*/, FullMask);
  assert(res4 == (threadIdx.x ^ 2) && pred4);
#endif // __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
}

__host__ __device__ void test_shfl_full_mask_no_pred()
{
#if __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
  constexpr unsigned FullMask = 0xFFFFFFFF;
  auto data                   = threadIdx.x;
  auto res1                   = cuda::ptx::shfl_sync_idx(data, 2 /*idx*/, 0b11111 /*clamp*/, FullMask);
  assert(res1 == 2);

  auto res2 = cuda::ptx::shfl_sync_up(data, 2 /*offset*/, 0 /*clamp*/, FullMask);
  if (threadIdx.x <= 1)
  {
    assert(res2 == threadIdx.x);
  }
  else
  {
    assert(res2 == threadIdx.x - 2);
  }

  auto res3 = cuda::ptx::shfl_sync_down(data, 2 /*offset*/, 0b11111 /*clamp*/, FullMask);
  if (threadIdx.x >= 30)
  {
    assert(res3 == threadIdx.x);
  }
  else
  {
    assert(res3 == threadIdx.x + 2);
  }

  auto res4 = cuda::ptx::shfl_sync_bfly(data, 2 /*offset*/, 0b11111 /*clamp*/, FullMask);
  assert(res4 == (threadIdx.x ^ 2));
#endif // __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
}

__host__ __device__ void test_shfl_partial_mask()
{
#if __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
  constexpr unsigned PartialMask = 0b1111;
  auto data                      = threadIdx.x;
  bool pred1;
  if (threadIdx.x <= 3)
  {
    auto res1 = cuda::ptx::shfl_sync_idx(data, pred1, 2 /*idx*/, 0b11111 /*clamp*/, PartialMask);
    assert(res1 == 2 && pred1);
  }
#endif // __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
}

__host__ __device__ void test_shfl_partial_warp()
{
#if __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
  constexpr unsigned FullMask = 0xFFFFFFFF;
  unsigned max_lane_mask      = 16;
  unsigned clamp              = 0b11111;
  unsigned clamp_segmark      = (max_lane_mask << 8) | clamp;
  auto data                   = threadIdx.x;
  bool pred1, pred2, pred3, pred4;
  auto res1 = cuda::ptx::shfl_sync_idx(data, pred1, 2 /*idx*/, clamp_segmark, FullMask);
  if (threadIdx.x < 16)
  {
    assert(res1 == 2 && pred1);
  }
  else
  {
    assert(res1 == 16 + 2 && pred1);
  }

  auto res2 = cuda::ptx::shfl_sync_up(data, pred2, 2 /*offset*/, (max_lane_mask << 8), FullMask);
  if (threadIdx.x <= 1 || threadIdx.x == 16 || threadIdx.x == 17)
  {
    assert(res2 == threadIdx.x && !pred2);
  }
  else
  {
    assert(res2 == threadIdx.x - 2 && pred2);
  }

  auto res3 = cuda::ptx::shfl_sync_down(data, pred3, 2 /*offset*/, clamp_segmark, FullMask);
  if (threadIdx.x == 14 || threadIdx.x == 15 || threadIdx.x >= 30)
  {
    assert(res3 == threadIdx.x && !pred3);
  }
  else
  {
    assert(res3 == threadIdx.x + 2 && pred3);
  }

  auto res4 = cuda::ptx::shfl_sync_bfly(data, pred4, 2 /*offset*/, clamp_segmark, FullMask);
  assert(res4 == (threadIdx.x ^ 2) && pred4);
#endif // __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
}

__host__ __device__ void test_shfl_divergence()
{
#if __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
  if (threadIdx.x == 0)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(10000000u);))
  }
  auto value = cuda::ptx::shfl_sync_idx(threadIdx.x, 0, 0b11111, 0xFFFFFFFF);
  assert(value == 0);
#endif // __cccl_ptx_isa >= 600 && _CCCL_DEVICE_COMPILATION()
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, cuda_thread_count = 32;)
  test_shfl_full_mask();
  test_shfl_partial_mask();
  test_shfl_partial_warp();
  NV_IF_TARGET(NV_PROVIDES_SM_70, (test_shfl_divergence();))
  return 0;
}
