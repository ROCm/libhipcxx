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

#ifndef _CUDA_PTX_BMSK_H_
#define _CUDA_PTX_BMSK_H_

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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_PTX

#include <cuda/__ptx/instructions/generated/bmsk.h>

#if _CCCL_HIP_COMPILATION()
// NOTE(HIP/AMD): Software emulation of PTX SM_70+ `bmsk.{clamp,wrap}.b32`,
// which builds a 32-bit mask of `b` consecutive set bits starting at bit
// position `a` (mod 32). The clamp variant truncates the run when it would
// cross bit 31; the wrap variant wraps around to bit 0. Pure arithmetic,
// no AMDGCN intrinsic needed. The upstream `__cccl_ptx_isa` for HIP is 0
// so the generated PTX bodies above are hidden; we provide HIP equivalents
// here in the same cuda::ptx namespace. Marked `_CCCL_API constexpr` so
// the implementation can be parity-verified against hand-computed PTX-
// manual reference values via `static_assert` (see
// `test/libcudacxx/cuda/ptx/ptx.hip_emu_bmsk_parity.compile.pass.cpp`).
// The upstream NV wrappers stay non-constexpr (inline asm); the HIP arm
// being more permissive does not break call-site compatibility.
template <typename = void>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::uint32_t
bmsk_clamp(::cuda::std::uint32_t __a_reg, ::cuda::std::uint32_t __b_reg) noexcept
{
  const ::cuda::std::uint32_t __start = __a_reg & 31u;
  // Clamp the run-length so the end position never exceeds bit 31.
  ::cuda::std::uint32_t __end = __start + __b_reg;
  if (__end > 32u)
  {
    __end = 32u;
  }
  if (__end <= __start)
  {
    return 0u;
  }
  const ::cuda::std::uint32_t __count = __end - __start;
  if (__count >= 32u)
  {
    return ~::cuda::std::uint32_t{0u};
  }
  return ((::cuda::std::uint32_t{1u} << __count) - 1u) << __start;
}

template <typename = void>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::uint32_t
bmsk_wrap(::cuda::std::uint32_t __a_reg, ::cuda::std::uint32_t __b_reg) noexcept
{
  if (__b_reg == 0u)
  {
    return 0u;
  }
  if (__b_reg >= 32u)
  {
    return ~::cuda::std::uint32_t{0u};
  }
  const ::cuda::std::uint32_t __start = __a_reg & 31u;
  const ::cuda::std::uint32_t __mask  = (::cuda::std::uint32_t{1u} << __b_reg) - 1u;
  if (__start == 0u)
  {
    return __mask;
  }
  // Rotate-left the b-bit run so it wraps around bit 0 if it overflows.
  return (__mask << __start) | (__mask >> (32u - __start));
}
#endif // _CCCL_HIP_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_BMSK_H_
