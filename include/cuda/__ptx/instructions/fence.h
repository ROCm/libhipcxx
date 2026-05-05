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

#ifndef _CUDA_PTX_FENCE_H_
#define _CUDA_PTX_FENCE_H_

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

// 9.7.12.4. Parallel Synchronization and Communication Instructions: membar/fence
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence
#include <cuda/__ptx/instructions/generated/fence.h>
#include <cuda/__ptx/instructions/generated/fence_mbarrier_init.h>
#include <cuda/__ptx/instructions/generated/fence_proxy_alias.h>
#include <cuda/__ptx/instructions/generated/fence_proxy_async.h>
#include <cuda/__ptx/instructions/generated/fence_proxy_async_generic_sync_restrict.h>
#include <cuda/__ptx/instructions/generated/fence_proxy_tensormap_generic.h>
#include <cuda/__ptx/instructions/generated/fence_sync_restrict.h>

#if _CCCL_HIP_COMPILATION()
// NOTE(HIP/AMD): Software equivalents of PTX
// `fence.{sc,acq_rel,acquire,release}.{cta,gpu,sys}`. Each maps to a
// single `__builtin_amdgcn_fence(<order>, <scope-string>)`. The scope
// strings are HIP/AMDGCN-specific:
//   PTX scope    -> AMDGCN scope string
//   .cta         -> "workgroup"
//   .gpu         -> "agent"
//   .sys         -> ""               (empty == system)
// PTX scope `.cluster` has no AMD equivalent (no thread-block clusters
// on AMDGCN); not exposed. The proxy fences (fence_proxy_*,
// fence_mbarrier_init, fence_sync_restrict, ...) are tied to NVIDIA-
// only proxies (TMA, tensormap, mbarrier) and stay NVIDIA-only.
// Ported from upgrade/3.1_base PTX-on-HIP roadmap
// (feat/moberste/add_partial_ptx_support_3_1).

#  if defined(__HIP_DEVICE_COMPILE__)
#    define _LIBHIPCXX_PTX_HIP_FENCE_BODY(__order)                                                            \
      static_assert(_Scope == dot_scope::cta || _Scope == dot_scope::gpu || _Scope == dot_scope::sys,         \
                    "fence(sem, scope): scope must be cta, gpu or sys on HIP (no .cluster on AMDGCN)");       \
      if constexpr (_Scope == dot_scope::cta)                                                                 \
      {                                                                                                       \
        __builtin_amdgcn_fence((__order), "workgroup");                                                       \
      }                                                                                                       \
      else if constexpr (_Scope == dot_scope::gpu)                                                            \
      {                                                                                                       \
        __builtin_amdgcn_fence((__order), "agent");                                                           \
      }                                                                                                       \
      else                                                                                                    \
      {                                                                                                       \
        __builtin_amdgcn_fence((__order), "");                                                                \
      }
#  else
#    define _LIBHIPCXX_PTX_HIP_FENCE_BODY(__order) ((void) 0)
#  endif

template <dot_scope _Scope>
_CCCL_DEVICE static inline void fence(sem_sc_t, scope_t<_Scope>)
{
  _LIBHIPCXX_PTX_HIP_FENCE_BODY(__ATOMIC_SEQ_CST);
}

template <dot_scope _Scope>
_CCCL_DEVICE static inline void fence(sem_acq_rel_t, scope_t<_Scope>)
{
  _LIBHIPCXX_PTX_HIP_FENCE_BODY(__ATOMIC_ACQ_REL);
}

template <dot_scope _Scope>
_CCCL_DEVICE static inline void fence(sem_acquire_t, scope_t<_Scope>)
{
  _LIBHIPCXX_PTX_HIP_FENCE_BODY(__ATOMIC_ACQUIRE);
}

template <dot_scope _Scope>
_CCCL_DEVICE static inline void fence(sem_release_t, scope_t<_Scope>)
{
  _LIBHIPCXX_PTX_HIP_FENCE_BODY(__ATOMIC_RELEASE);
}

#  undef _LIBHIPCXX_PTX_HIP_FENCE_BODY
#endif // _CCCL_HIP_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_FENCE_H_
