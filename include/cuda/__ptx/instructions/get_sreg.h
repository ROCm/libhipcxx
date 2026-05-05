// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _CUDA_PTX_GET_SREG_H_
#define _CUDA_PTX_GET_SREG_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/ptx_dot_variants.h>
#include <cuda/__ptx/ptx_helper_functions.h>
#include <cuda/std/cstdint>

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

// NOTE(HIP/AMD): include <amd/amd_utils.h> for _CCCL_HIP_WAVE_SIZE used
// by the HIP-side lanemask emulations below.
#if _CCCL_HIP_COMPILATION()
#  include <amd/amd_utils.h>
#endif

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_PTX

// 10. Special Registers
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers
#include <cuda/__ptx/instructions/generated/get_sreg.h>

#if _CCCL_HIP_COMPILATION()
// NOTE(HIP/AMD): Software emulations of the warp-lane special registers
// (%laneid, %lanemask_{eq,le,lt,ge,gt}) and the block / grid coordinate
// SREGs (%tid.*, %ntid.*, %ctaid.*, %nctaid.*) and cycle counters
// (%clock, %clock_hi, %clock64). The other PTX special registers
// (%smid, %cluster_*, %globaltimer, ...) are not exposed on HIP. Ported
// from upgrade/3.1_base PTX-on-HIP roadmap
// (feat/moberste/add_partial_ptx_support_3_1).
//
// Wave size compatibility:
//   PTX `%lanemask_*` SREGs are 32-bit because NVIDIA warps are 32 lanes
//   wide. AMD wavefronts are 32 lanes on gfx10+/RDNA but 64 lanes on
//   gfx9 (MI100/MI200/MI300). To stay correct on both *without* the
//   kernel signature changing between host and device compilation
//   passes (which would break HIP kernel-symbol registration), the HIP
//   lanemask return type is `unsigned long long` (uint64_t)
//   unconditionally:
//     - on wave-32 the high 32 bits are always 0 (no information loss),
//     - on wave-64 all 64 bits are used (full correctness).
//   The host pass agrees with both wave-32 and wave-64 device passes
//   on this type. Use `auto` at the call site to remain wave-size-
//   portable; capturing into a `uint32_t` on wave-64 would silently
//   drop lanes 32..63.

using __hip_lanemask_t = unsigned long long;

#  if _CCCL_HIP_WAVE_SIZE >= 64
inline constexpr __hip_lanemask_t __hip_wave_mask = ~__hip_lanemask_t{0};
#  else
inline constexpr __hip_lanemask_t __hip_wave_mask = (__hip_lanemask_t{1} << _CCCL_HIP_WAVE_SIZE) - 1u;
#  endif

template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint32_t get_sreg_laneid()
{
#  if defined(__HIP_DEVICE_COMPILE__)
  // Returns 0..warpSize-1; on wave-64 this can exceed 31, unlike PTX %laneid.
  return ::__lane_id();
#  else
  return 0u;
#  endif
}

template <typename = void>
_CCCL_DEVICE static inline __hip_lanemask_t get_sreg_lanemask_eq()
{
#  if defined(__HIP_DEVICE_COMPILE__)
  return __hip_lanemask_t{1} << ::__lane_id();
#  else
  return __hip_lanemask_t{0};
#  endif
}

template <typename = void>
_CCCL_DEVICE static inline __hip_lanemask_t get_sreg_lanemask_lt()
{
#  if defined(__HIP_DEVICE_COMPILE__)
  return (__hip_lanemask_t{1} << ::__lane_id()) - __hip_lanemask_t{1};
#  else
  return __hip_lanemask_t{0};
#  endif
}

template <typename = void>
_CCCL_DEVICE static inline __hip_lanemask_t get_sreg_lanemask_le()
{
#  if defined(__HIP_DEVICE_COMPILE__)
  // Special-case the highest lane to avoid `1ULL << 64` UB on wave-64
  // and to keep positions 32..63 zero on wave-32.
  const unsigned __l = ::__lane_id();
  return (__l == _CCCL_HIP_WAVE_SIZE - 1u)
           ? __hip_wave_mask
           : ((__hip_lanemask_t{1} << (__l + 1u)) - __hip_lanemask_t{1});
#  else
  return __hip_lanemask_t{0};
#  endif
}

template <typename = void>
_CCCL_DEVICE static inline __hip_lanemask_t get_sreg_lanemask_ge()
{
#  if defined(__HIP_DEVICE_COMPILE__)
  return (~((__hip_lanemask_t{1} << ::__lane_id()) - __hip_lanemask_t{1})) & __hip_wave_mask;
#  else
  return __hip_lanemask_t{0};
#  endif
}

template <typename = void>
_CCCL_DEVICE static inline __hip_lanemask_t get_sreg_lanemask_gt()
{
#  if defined(__HIP_DEVICE_COMPILE__)
  const unsigned __l = ::__lane_id();
  return (__l == _CCCL_HIP_WAVE_SIZE - 1u)
           ? __hip_lanemask_t{0}
           : ((~((__hip_lanemask_t{1} << (__l + 1u)) - __hip_lanemask_t{1})) & __hip_wave_mask);
#  else
  return __hip_lanemask_t{0};
#  endif
}

// Block / grid coordinate SREGs. Map to standard HIP/CUDA built-in
// `threadIdx`, `blockDim`, `blockIdx`, `gridDim`. Only meaningful at
// device runtime; on the host pass we return 0 as a stub. PTX SREGs
// return uint32_t; HIP `threadIdx.x` is `unsigned`. Same width.
#  define _LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(__name, __expr)                          \
    template <typename = void>                                                           \
    _CCCL_DEVICE static inline ::cuda::std::uint32_t __name()                            \
    {                                                                                    \
      return (__expr);                                                                   \
    }

#  if defined(__HIP_DEVICE_COMPILE__)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_tid_x, threadIdx.x)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_tid_y, threadIdx.y)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_tid_z, threadIdx.z)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ntid_x, blockDim.x)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ntid_y, blockDim.y)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ntid_z, blockDim.z)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ctaid_x, blockIdx.x)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ctaid_y, blockIdx.y)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ctaid_z, blockIdx.z)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_nctaid_x, gridDim.x)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_nctaid_y, gridDim.y)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_nctaid_z, gridDim.z)
#  else
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_tid_x, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_tid_y, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_tid_z, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ntid_x, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ntid_y, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ntid_z, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ctaid_x, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ctaid_y, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_ctaid_z, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_nctaid_x, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_nctaid_y, 0u)
_LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG(get_sreg_nctaid_z, 0u)
#  endif

#  undef _LIBHIPCXX_PTX_DEFINE_BLOCK_GRID_SREG

// Cycle counters. Map to HIP `clock()` / `clock64()`.
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint32_t get_sreg_clock()
{
#  if defined(__HIP_DEVICE_COMPILE__)
  return static_cast<::cuda::std::uint32_t>(::clock());
#  else
  return 0u;
#  endif
}

template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint32_t get_sreg_clock_hi()
{
#  if defined(__HIP_DEVICE_COMPILE__)
  return static_cast<::cuda::std::uint32_t>(static_cast<::cuda::std::uint64_t>(::clock64()) >> 32);
#  else
  return 0u;
#  endif
}

template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint64_t get_sreg_clock64()
{
#  if defined(__HIP_DEVICE_COMPILE__)
  return static_cast<::cuda::std::uint64_t>(::clock64());
#  else
  return 0ull;
#  endif
}
#endif // _CCCL_HIP_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_GET_SREG_H_
