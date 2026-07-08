//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

// <cuda/ptx>

// NOTE(HIP/AMD): on HIP the public umbrella <cuda/ptx> hard-errors
// (cuda::ptx is an NV-only public API per the design); route through
// the individual instruction header instead, which carries the HIP
// software-emulated implementation behind a _CCCL_HIP_COMPILATION()
// gate.
#ifdef __HIP_PLATFORM_AMD__
#  include <cuda/__ptx/instructions/fence.h>
#else
#  include <cuda/ptx>
#endif
#include <cuda/std/utility>

#include "generated/fence.h"
#include "generated/fence_mbarrier_init.h"
#include "generated/fence_proxy_alias.h"
#include "generated/fence_proxy_async.h"
#include "generated/fence_proxy_async_generic_sync_restrict.h"
#include "generated/fence_proxy_tensormap_generic.h"
#include "generated/fence_sync_restrict.h"

// NOTE(HIP/AMD): the upstream fence-overload instantiations in
// 'generated/fence.h' are gated on '__cccl_ptx_isa >= 600 &&
// NV_PROVIDES_SM_70' (false on HIP). Force HIP overload instantiation
// explicitly so this compile.pass.cpp exercises the HIP wrappers in
// <cuda/__ptx/instructions/fence.h> -- 4 sems (sc/acq_rel/acquire/
// release) x 3 scopes (cta/gpu/sys) = 12 overloads, each mapped to a
// '__builtin_amdgcn_fence(<order>, <scope-string>)' call. The proxy
// fences (fence_proxy_*, fence_mbarrier_init, fence_sync_restrict) are
// NV-only (tied to NV-specific proxies: TMA, mbarrier, tensormap) and
// are not emulated on HIP -- their fn_ptr blocks in the included
// 'generated/fence_proxy_*.h' headers stay gated on __cccl_ptx_isa
// and compile to nothing here.
#if _CCCL_HIP_COMPILATION()
__global__ void test_fence_hip(void** __fn_ptr)
{
  // .cta scope
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
  // .gpu scope
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
  // .sys scope
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));
  *__fn_ptr++ = reinterpret_cast<void*>(
    static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));
}
#endif // _CCCL_HIP_COMPILATION()

int main(int, char**)
{
  return 0;
}
