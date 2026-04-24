//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from 7011c55229 (9e9eeeb439) - COMMENTED OUT
// // MIT License
// //
// // Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// //
// // Permission is hereby granted, free of charge, to any person obtaining a copy
// // of this software and associated documentation files (the "Software"), to deal
// // in the Software without restriction, including without limitation the rights
// // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // copies of the Software, and to permit persons to whom the Software is
// // furnished to do so, subject to the following conditions:
// //
// // The above copyright notice and this permission notice shall be included in all
// // copies or substantial portions of the Software.
// //
// // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// // SOFTWARE.
//
// =======
// >>>>>>> END NEW CODE (9e9eeeb439)
// UNSUPPORTED: libcpp-has-no-threads

// <cuda/ptx>

// NOTE(HIP/AMD): on HIP the public umbrella <cuda/ptx> hard-errors
// (cuda::ptx is an NV-only public API per the design); route through
// the individual instruction header instead, which carries the HIP
// software-emulated implementation behind a _CCCL_HIP_COMPILATION()
// gate.
#ifdef __HIP_PLATFORM_AMD__
#  include <cuda/__ptx/instructions/elect_sync.h>
#else
#  include <cuda/ptx>
#endif
#include <cuda/std/utility>

#include "generated/elect_sync.h"

// NOTE(HIP/AMD): the upstream 'generated/elect_sync.h' fn_ptr
// instantiation is gated on '__cccl_ptx_isa >= 800 && NV_PROVIDES_SM_90'
// (false on HIP). Force HIP overload instantiation explicitly so this
// compile.pass.cpp exercises the HIP wrapper in
// <cuda/__ptx/instructions/elect_sync.h>. The HIP arm is implemented
// unconditionally (no SM-version gate); semantic is "elect the lowest
// active lane in (membermask & __activemask())". On wave-64 lanes
// 32..63 are filtered out by the AND with membermask (uint32_t),
// preserving PTX 32-lane warp semantics.
#if _CCCL_HIP_COMPILATION()
__global__ void test_elect_sync_hip(void** __fn_ptr)
{
  *__fn_ptr++ =
    reinterpret_cast<void*>(static_cast<bool (*)(const cuda::std::uint32_t&)>(cuda::ptx::elect_sync));
}
#endif // _CCCL_HIP_COMPILATION()

int main(int, char**)
{
  return 0;
}
