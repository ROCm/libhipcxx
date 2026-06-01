// -*- C++ -*-
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

#ifndef _CUDA_PTX_ELECT_SYNC_H_
#define _CUDA_PTX_ELECT_SYNC_H_

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

#include <cuda/__ptx/instructions/generated/elect_sync.h>

#if _CCCL_HIP_COMPILATION()
// NOTE(HIP/AMD): EXPERIMENTAL -- HIP emulation for specific in-tree
// consumers only; not bit-exact-verified vs the NVPTX implementation and
// may change. The public <cuda/ptx> surface stays unsupported on HIP (it
// #errors); see the consolidated NOTE in <cuda/ptx>.
// NOTE(HIP/AMD): Software emulation of PTX SM_90+ `elect.sync`. Returns
// true on exactly the lane corresponding to the lowest set bit of
// (membermask & __activemask()). Slower than the dedicated PTX
// instruction.
template <typename = void>
_CCCL_DEVICE static inline bool elect_sync(const ::cuda::std::uint32_t& __membermask)
{
#  if defined(__HIP_DEVICE_COMPILE__)
  const unsigned long long __active = static_cast<unsigned long long>(__membermask) & ::__activemask();
  if (__active == 0ull)
  {
    return false;
  }
  // __ffsll is 1-based; subtract 1 for 0-based lane id.
  return static_cast<unsigned>(::__ffsll(static_cast<long long>(__active)) - 1) == ::__lane_id();
#  else
  // On the host pass the function is never called; just return false.
  (void) __membermask;
  return false;
#  endif
}
#endif // _CCCL_HIP_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_ELECT_SYNC_H_
