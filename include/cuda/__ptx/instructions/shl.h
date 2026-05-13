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

#ifndef _CUDA_PTX_SHL_H_
#define _CUDA_PTX_SHL_H_

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
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/cstdint>

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_PTX

#include <cuda/__ptx/instructions/generated/shl.h>

#if _CCCL_HIP_COMPILATION()
// NOTE(HIP/AMD): Software emulation of PTX `shl.b{16,32,64}` (logical
// left shift). PTX clamps the shift amount to the bit-width modulo
// behaviour: shift amounts >= width return 0. Mirror the upstream
// _B16/_B32/_B64 template-by-size pattern so the same call sites work
// on both NVIDIA and HIP. Pure C++ arithmetic, no AMDGCN intrinsic
// needed. The upstream `__cccl_ptx_isa` for HIP is 0 so the generated
// PTX bodies above are hidden; we provide HIP equivalents here in the
// same cuda::ptx namespace.
// NOTE(HIP/AMD): convert via cuda::std::bit_cast (well-defined memcpy-style
// punning) instead of '*reinterpret_cast<...*>' which would alias incompatible
// types and is undefined behaviour under strict aliasing.
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 shl(_B16 __a_reg, ::cuda::std::uint32_t __b_reg)
{
  const auto __a_bits = ::cuda::std::bit_cast<::cuda::std::uint16_t>(__a_reg);
  const ::cuda::std::uint16_t __dest =
    (__b_reg >= 16u) ? ::cuda::std::uint16_t{0u} : static_cast<::cuda::std::uint16_t>(__a_bits << __b_reg);
  return ::cuda::std::bit_cast<_B16>(__dest);
}

template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 shl(_B32 __a_reg, ::cuda::std::uint32_t __b_reg)
{
  const auto __a_bits = ::cuda::std::bit_cast<::cuda::std::uint32_t>(__a_reg);
  const ::cuda::std::uint32_t __dest = (__b_reg >= 32u) ? ::cuda::std::uint32_t{0u} : (__a_bits << __b_reg);
  return ::cuda::std::bit_cast<_B32>(__dest);
}

template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 shl(_B64 __a_reg, ::cuda::std::uint32_t __b_reg)
{
  const auto __a_bits = ::cuda::std::bit_cast<::cuda::std::uint64_t>(__a_reg);
  const ::cuda::std::uint64_t __dest = (__b_reg >= 64u) ? ::cuda::std::uint64_t{0u} : (__a_bits << __b_reg);
  return ::cuda::std::bit_cast<_B64>(__dest);
}
#endif // _CCCL_HIP_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_SHL_H_
