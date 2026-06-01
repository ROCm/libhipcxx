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

#ifndef _CUDA_PTX_SHFL_SYNC_H
#define _CUDA_PTX_SHFL_SYNC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__ptx/ptx_dot_variants.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/cstdint>

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_PTX

#if __cccl_ptx_isa >= 600

enum class __dot_shfl_mode
{
  __up,
  __down,
  __bfly,
  __idx
};

[[maybe_unused]]
_CCCL_DEVICE static inline uint32_t
__shfl_sync_dst_lane(__dot_shfl_mode __shfl_mode, uint32_t __lane_idx_offset, uint32_t __clamp_segmask)
{
  auto __lane     = ::cuda::ptx::get_sreg_laneid();
  auto __clamp    = __clamp_segmask & 0b11111;
  auto __segmask  = __clamp_segmask >> 8;
  auto __max_lane = (__lane & __segmask) | (__clamp & ~__segmask);
  uint32_t __j    = 0;
  if (__shfl_mode == __dot_shfl_mode::__idx)
  {
    auto __min_lane = __lane & __segmask;
    __j             = __min_lane | (__lane_idx_offset & ~__segmask);
  }
  else if (__shfl_mode == __dot_shfl_mode::__up)
  {
    __j = __lane_idx_offset >= __lane ? 0 : __lane - __lane_idx_offset;
  }
  else if (__shfl_mode == __dot_shfl_mode::__down)
  {
    __j = __lane + __lane_idx_offset;
  }
  else
  {
    __j = __lane ^ __lane_idx_offset;
  }
  auto __dst = __shfl_mode == __dot_shfl_mode::__up
               ? (__j >= __max_lane ? __j : __lane) //
               : (__j <= __max_lane ? __j : __lane);
  return (1u << __dst);
}

template <typename _Tp>
_CCCL_DEVICE static inline void __shfl_sync_checks(
  __dot_shfl_mode __shfl_mode,
  _Tp,
  [[maybe_unused]] uint32_t __lane_idx_offset,
  [[maybe_unused]] uint32_t __clamp_segmask,
  [[maybe_unused]] uint32_t __lane_mask)
{
  static_assert(sizeof(_Tp) == 4, "shfl.sync only accepts 4-byte data types");
  _CCCL_ASSERT(__lane_mask & (1u << ::cuda::ptx::get_sreg_laneid()), "lane_mask must contain the current lane");
  if (__shfl_mode != __dot_shfl_mode::__idx)
  {
    _CCCL_ASSERT(__lane_idx_offset < 32, "the lane index or offset must be less than the warp size");
  }
  _CCCL_ASSERT((__clamp_segmask | 0b1111100011111) == 0b1111100011111,
               "clamp value + segmentation mask must use the bit positions [0:4] and [8:12]");
  _CCCL_ASSERT(::cuda::ptx::__shfl_sync_dst_lane(__shfl_mode, __lane_idx_offset, __clamp_segmask) & __lane_mask,
               "the destination lane must be a member of the lane mask");
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_idx(
  _Tp __data, bool& __pred, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  ::cuda::ptx::__shfl_sync_checks(__dot_shfl_mode::__idx, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = ::cuda::std::bit_cast<uint32_t>(__data);
  int __pred1;
  uint32_t __ret;
  asm volatile(
    "{                                                      \n\t\t"
    ".reg .pred p;                                          \n\t\t"
    "shfl.sync.idx.b32 %0|p, %2, %3, %4, %5;                \n\t\t"
    "selp.s32 %1, 1, 0, p;                                  \n\t"
    "}"
    : "=r"(__ret), "=r"(__pred1)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  __pred = static_cast<bool>(__pred1);
  return ::cuda::std::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_idx(_Tp __data, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  ::cuda::ptx::__shfl_sync_checks(__dot_shfl_mode::__idx, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = ::cuda::std::bit_cast<uint32_t>(__data);
  uint32_t __ret;
  asm volatile("{                                                      \n\t\t"
               "shfl.sync.idx.b32 %0, %1, %2, %3, %4;                  \n\t\t"
               "}"
               : "=r"(__ret)
               : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  return ::cuda::std::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_up(
  _Tp __data, bool& __pred, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  ::cuda::ptx::__shfl_sync_checks(__dot_shfl_mode::__up, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = ::cuda::std::bit_cast<uint32_t>(__data);
  int __pred1;
  uint32_t __ret;
  asm volatile(
    "{                                                      \n\t\t"
    ".reg .pred p;                                          \n\t\t"
    "shfl.sync.up.b32 %0|p, %2, %3, %4, %5;                 \n\t\t"
    "selp.s32 %1, 1, 0, p;                                  \n\t"
    "}"
    : "=r"(__ret), "=r"(__pred1)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  __pred = static_cast<bool>(__pred1);
  return ::cuda::std::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_up(_Tp __data, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  ::cuda::ptx::__shfl_sync_checks(__dot_shfl_mode::__up, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = ::cuda::std::bit_cast<uint32_t>(__data);
  uint32_t __ret;
  asm volatile("{                                                      \n\t\t"
               "shfl.sync.up.b32 %0, %1, %2, %3, %4;                   \n\t\t"
               "}"
               : "=r"(__ret)
               : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  return ::cuda::std::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_down(
  _Tp __data, bool& __pred, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  ::cuda::ptx::__shfl_sync_checks(__dot_shfl_mode::__down, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = ::cuda::std::bit_cast<uint32_t>(__data);
  int __pred1;
  uint32_t __ret;
  asm volatile(
    "{                                                      \n\t\t"
    ".reg .pred p;                                          \n\t\t"
    "shfl.sync.down.b32 %0|p, %2, %3, %4, %5;               \n\t\t"
    "selp.s32 %1, 1, 0, p;                                  \n\t"
    "}"
    : "=r"(__ret), "=r"(__pred1)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  __pred = static_cast<bool>(__pred1);
  return ::cuda::std::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_down(_Tp __data, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  ::cuda::ptx::__shfl_sync_checks(__dot_shfl_mode::__down, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = ::cuda::std::bit_cast<uint32_t>(__data);
  uint32_t __ret;
  asm volatile("{                                                      \n\t\t"
               "shfl.sync.down.b32 %0, %1, %2, %3, %4;                 \n\t\t"
               "}"
               : "=r"(__ret)
               : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  return ::cuda::std::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_bfly(
  _Tp __data, bool& __pred, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  ::cuda::ptx::__shfl_sync_checks(__dot_shfl_mode::__bfly, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = ::cuda::std::bit_cast<uint32_t>(__data);
  int __pred1;
  uint32_t __ret;
  asm volatile(
    "{                                                      \n\t\t"
    ".reg .pred p;                                          \n\t\t"
    "shfl.sync.bfly.b32 %0|p, %2, %3, %4, %5;               \n\t\t"
    "selp.s32 %1, 1, 0, p;                                  \n\t"
    "}"
    : "=r"(__ret), "=r"(__pred1)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  __pred = static_cast<bool>(__pred1);
  return ::cuda::std::bit_cast<uint32_t>(__ret);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_bfly(_Tp __data, uint32_t __lane_idx_offset, uint32_t __clamp_segmask, uint32_t __lane_mask) noexcept
{
  ::cuda::ptx::__shfl_sync_checks(__dot_shfl_mode::__bfly, __data, __lane_idx_offset, __clamp_segmask, __lane_mask);
  auto __data1 = ::cuda::std::bit_cast<uint32_t>(__data);
  uint32_t __ret;
  asm volatile( //
    "{                                                      \n\t\t"
    "shfl.sync.bfly.b32 %0, %1, %2, %3, %4;                 \n\t\t"
    "}"
    : "=r"(__ret)
    : "r"(__data1), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  return ::cuda::std::bit_cast<uint32_t>(__ret);
}

#endif // __cccl_ptx_isa >= 600

#if _CCCL_HIP_COMPILATION()
// NOTE(HIP/AMD): EXPERIMENTAL -- HIP emulation for specific in-tree
// consumers only; not bit-exact-verified vs the NVPTX implementation and
// may change. The public <cuda/ptx> surface stays unsupported on HIP (it
// #errors); see the consolidated NOTE in <cuda/ptx>.
// NOTE(HIP/AMD): Software equivalents of PTX `shfl.sync.{idx,up,down,bfly}.b32`,
// implemented on top of HIP's `__shfl{,_up,_down,_xor}` family.
//
// PTX <-> HIP signature mapping notes:
//   1. lane_mask: PTX requires the caller to pass an explicit bitmask
//      of participating lanes; HIP `__shfl*` is implicitly synchronized
//      across the active wave. The lane_mask argument is therefore
//      *ignored* on HIP. Code that relied on lanes outside the mask
//      not participating must be audited; on AMDGCN the underlying
//      `ds_swizzle`/`ds_bpermute` always involves all active lanes.
//   2. clamp_segmask: PTX packs `clamp:segmask` into one uint32_t (low
//      5 bits = clamp, bits 8..12 = segmask). HIP's third "width"
//      parameter has the same role for sub-warp segments. We extract
//      the segmask, derive the segment width, and pass it through.
//   3. wave size: PTX `shfl.sync` is defined for 32-wide warps only.
//      On AMD wave-64 (gfx9) the same call shuffles across 64 lanes.
//      The `width` parameter caps the effective shuffle width and is
//      honoured by the HIP shuffle family on both wave-32 and wave-64.
//
// PTX `shfl.sync.b32` only accepts 4-byte data; mirrored with a
// static_assert. The `bool& pred` overloads return whether the source
// lane was active. On HIP all participating lanes are by definition
// active, so we always set pred to true.

namespace __detail
{
[[maybe_unused]]
_CCCL_DEVICE static inline int __hip_shfl_width_from_clamp_segmask(::cuda::std::uint32_t __clamp_segmask) noexcept
{
  // PTX clamp_segmask layout: low 5 bits = clamp position, bits 8..12 =
  // segmask. The segmask `0b11111` (31) means "no segmentation"
  // (whole warp); a smaller segmask defines sub-warps of size
  // 32 - segmask. HIP `__shfl*` width is the segment size in lanes.
  const ::cuda::std::uint32_t __segmask = (__clamp_segmask >> 8) & 0x1f;
  if (__segmask == 0x1f)
  {
#  if defined(__HIP_DEVICE_COMPILE__)
    return static_cast<int>(warpSize);
#  else
    return 32;
#  endif
  }
  const int __seg = static_cast<int>(32u - __segmask);
  return __seg < 2 ? 2 : __seg;
}
} // namespace __detail

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_idx(
  _Tp __data, bool& __pred, [[maybe_unused]] ::cuda::std::uint32_t __lane_idx_offset,
  [[maybe_unused]] ::cuda::std::uint32_t __clamp_segmask, ::cuda::std::uint32_t __lane_mask) noexcept
{
  static_assert(sizeof(_Tp) == 4, "shfl.sync only accepts 4-byte data types");
  (void) __lane_mask;
#  if defined(__HIP_DEVICE_COMPILE__)
  const int __width = __detail::__hip_shfl_width_from_clamp_segmask(__clamp_segmask);
  const auto __data_u = ::cuda::std::bit_cast<::cuda::std::uint32_t>(__data);
  const ::cuda::std::uint32_t __ret = ::__shfl(__data_u, static_cast<int>(__lane_idx_offset), __width);
  __pred = true;
  return ::cuda::std::bit_cast<_Tp>(__ret);
#  else
  __pred = false;
  return __data;
#  endif
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_idx(_Tp __data, ::cuda::std::uint32_t __lane_idx_offset,
              ::cuda::std::uint32_t __clamp_segmask, ::cuda::std::uint32_t __lane_mask) noexcept
{
  bool __pred_unused;
  return ::cuda::ptx::shfl_sync_idx(__data, __pred_unused, __lane_idx_offset, __clamp_segmask, __lane_mask);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_up(
  _Tp __data, bool& __pred, [[maybe_unused]] ::cuda::std::uint32_t __lane_idx_offset,
  [[maybe_unused]] ::cuda::std::uint32_t __clamp_segmask, ::cuda::std::uint32_t __lane_mask) noexcept
{
  static_assert(sizeof(_Tp) == 4, "shfl.sync only accepts 4-byte data types");
  (void) __lane_mask;
#  if defined(__HIP_DEVICE_COMPILE__)
  const int __width = __detail::__hip_shfl_width_from_clamp_segmask(__clamp_segmask);
  const auto __data_u = ::cuda::std::bit_cast<::cuda::std::uint32_t>(__data);
  const ::cuda::std::uint32_t __ret = ::__shfl_up(__data_u, __lane_idx_offset, __width);
  __pred = true;
  return ::cuda::std::bit_cast<_Tp>(__ret);
#  else
  __pred = false;
  return __data;
#  endif
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_up(_Tp __data, ::cuda::std::uint32_t __lane_idx_offset,
             ::cuda::std::uint32_t __clamp_segmask, ::cuda::std::uint32_t __lane_mask) noexcept
{
  bool __pred_unused;
  return ::cuda::ptx::shfl_sync_up(__data, __pred_unused, __lane_idx_offset, __clamp_segmask, __lane_mask);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_down(
  _Tp __data, bool& __pred, [[maybe_unused]] ::cuda::std::uint32_t __lane_idx_offset,
  [[maybe_unused]] ::cuda::std::uint32_t __clamp_segmask, ::cuda::std::uint32_t __lane_mask) noexcept
{
  static_assert(sizeof(_Tp) == 4, "shfl.sync only accepts 4-byte data types");
  (void) __lane_mask;
#  if defined(__HIP_DEVICE_COMPILE__)
  const int __width = __detail::__hip_shfl_width_from_clamp_segmask(__clamp_segmask);
  const auto __data_u = ::cuda::std::bit_cast<::cuda::std::uint32_t>(__data);
  const ::cuda::std::uint32_t __ret = ::__shfl_down(__data_u, __lane_idx_offset, __width);
  __pred = true;
  return ::cuda::std::bit_cast<_Tp>(__ret);
#  else
  __pred = false;
  return __data;
#  endif
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_down(_Tp __data, ::cuda::std::uint32_t __lane_idx_offset,
               ::cuda::std::uint32_t __clamp_segmask, ::cuda::std::uint32_t __lane_mask) noexcept
{
  bool __pred_unused;
  return ::cuda::ptx::shfl_sync_down(__data, __pred_unused, __lane_idx_offset, __clamp_segmask, __lane_mask);
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp shfl_sync_bfly(
  _Tp __data, bool& __pred, [[maybe_unused]] ::cuda::std::uint32_t __lane_idx_offset,
  [[maybe_unused]] ::cuda::std::uint32_t __clamp_segmask, ::cuda::std::uint32_t __lane_mask) noexcept
{
  static_assert(sizeof(_Tp) == 4, "shfl.sync only accepts 4-byte data types");
  (void) __lane_mask;
#  if defined(__HIP_DEVICE_COMPILE__)
  const int __width = __detail::__hip_shfl_width_from_clamp_segmask(__clamp_segmask);
  const auto __data_u = ::cuda::std::bit_cast<::cuda::std::uint32_t>(__data);
  const ::cuda::std::uint32_t __ret = ::__shfl_xor(__data_u, static_cast<int>(__lane_idx_offset), __width);
  __pred = true;
  return ::cuda::std::bit_cast<_Tp>(__ret);
#  else
  __pred = false;
  return __data;
#  endif
}

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE static inline _Tp
shfl_sync_bfly(_Tp __data, ::cuda::std::uint32_t __lane_idx_offset,
               ::cuda::std::uint32_t __clamp_segmask, ::cuda::std::uint32_t __lane_mask) noexcept
{
  bool __pred_unused;
  return ::cuda::ptx::shfl_sync_bfly(__data, __pred_unused, __lane_idx_offset, __clamp_segmask, __lane_mask);
}
#endif // _CCCL_HIP_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_SHFL_SYNC_H
