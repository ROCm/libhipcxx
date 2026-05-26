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
// UNSUPPORTED: clang && !nvcc

// <cuda/ptx>

// NOTE(HIP/AMD): on HIP the public umbrella <cuda/ptx> hard-errors
// (cuda::ptx is an NV-only public API per the design); route through
// the individual instruction header instead, which carries the HIP
// software-emulated implementation behind a _CCCL_HIP_COMPILATION()
// gate.
#ifdef __HIP_PLATFORM_AMD__
#  include <cuda/__ptx/instructions/get_sreg.h>
#else
#  include <cuda/ptx>
#endif
#include <cuda/std/utility>

#include "generated/get_sreg.h"

// NOTE(HIP/AMD): the upstream 'generated/get_sreg.h' fn_ptr
// instantiations are gated on '__cccl_ptx_isa >= N' (false on HIP).
// Force HIP overload instantiation explicitly so this compile.pass.cpp
// exercises the HIP wrappers in <cuda/__ptx/instructions/get_sreg.h>.
// Covers: laneid, lanemask_{eq,lt,le,ge,gt}, tid/ntid/ctaid/nctaid
// xyz, clock/clock_hi/clock64, total_smem_size. The lanemask family
// returns 'unsigned long long' (uint64_t) on HIP rather than uint32_t
// to remain wave-32 / wave-64 portable -- see the long NOTE in
// get_sreg.h for the rationale.
#if _CCCL_HIP_COMPILATION()
__global__ void test_get_sreg_hip(void** __fn_ptr)
{
  using __hip_lanemask_t = unsigned long long;
  // Lane + lanemask family
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_laneid));
  *__fn_ptr++ =
    reinterpret_cast<void*>(static_cast<__hip_lanemask_t (*)()>(cuda::ptx::get_sreg_lanemask_eq));
  *__fn_ptr++ =
    reinterpret_cast<void*>(static_cast<__hip_lanemask_t (*)()>(cuda::ptx::get_sreg_lanemask_lt));
  *__fn_ptr++ =
    reinterpret_cast<void*>(static_cast<__hip_lanemask_t (*)()>(cuda::ptx::get_sreg_lanemask_le));
  *__fn_ptr++ =
    reinterpret_cast<void*>(static_cast<__hip_lanemask_t (*)()>(cuda::ptx::get_sreg_lanemask_ge));
  *__fn_ptr++ =
    reinterpret_cast<void*>(static_cast<__hip_lanemask_t (*)()>(cuda::ptx::get_sreg_lanemask_gt));
  // Block + grid coordinates
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_tid_x));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_tid_y));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_tid_z));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_ntid_x));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_ntid_y));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_ntid_z));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_ctaid_x));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_ctaid_y));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_ctaid_z));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_nctaid_x));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_nctaid_y));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_nctaid_z));
  // Cycle counters
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_clock));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_clock_hi));
  *__fn_ptr++ = reinterpret_cast<void*>(static_cast<cuda::std::uint64_t (*)()>(cuda::ptx::get_sreg_clock64));
  // Shared (LDS) memory upper bound
  *__fn_ptr++ =
    reinterpret_cast<void*>(static_cast<cuda::std::uint32_t (*)()>(cuda::ptx::get_sreg_total_smem_size));
}
#endif // _CCCL_HIP_COMPILATION()

int main(int, char**)
{
  return 0;
}
