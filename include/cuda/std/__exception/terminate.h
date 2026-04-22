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

// Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef _CUDA_STD___EXCEPTION_TERMINATE_H
#define _CUDA_STD___EXCEPTION_TERMINATE_H

#include <cuda/std/detail/__config>
#include <libhipcxx/__amd/amd_utils.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// <<<<<<< OLD CODE from 7c099f6aec (10e65aca2b) - COMMENTED OUT
// // NOTE(HIP/AMD): Under hipRTC __cccl_terminate() takes the device branch
// // (libhipcxx::__trap()), so host <stdlib.h> (for ::exit) is unused. Including it
// // here would pull in host <stddef.h>/<bits/stdint-intn.h> on the JIT include
// // path and cause max_align_t / byte / int64_t redefinition conflicts. Skip it
// // under hipRTC, like NVRTC.
// #if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
// =======
#if _CCCL_TILE_COMPILATION()
#  include <cuda/std/cassert>
#endif // !_CCCL_TILE_COMPILATION()

#if !_CCCL_COMPILER(NVRTC)
// >>>>>>> END NEW CODE (10e65aca2b)
#  include <stdlib.h>
#endif // !_CCCL_COMPILER(NVRTC) && !_CCCL_COMPILER_HIPRTC

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code

_CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION // purposefully not using versioning namespace

[[noreturn]] _CCCL_API inline void __cccl_terminate() noexcept
{
// <<<<<<< OLD CODE from 7c099f6aec (10e65aca2b) - COMMENTED OUT
//   NV_IF_ELSE_TARGET(NV_IS_HOST, (::exit(-1);), (libhipcxx::__trap();))
// =======
#if _CCCL_TILE_COMPILATION()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::exit(-1);), (assert(false);))
#else // ^^^ _CCCL_TILE_COMPILATION() ^^^ / vvv !_CCCL_TILE_COMPILATION()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::exit(-1);), (::__trap();))
// >>>>>>> END NEW CODE (10e65aca2b)
  _CCCL_UNREACHABLE();
#endif // !_CCCL_TILE_COMPILATION()
}

#if 0 // Expose once atomic is universally available

using terminate_handler = void (*)();

#  ifdef __CUDA_ARCH__
__device__
#  endif // __CUDA_ARCH__
  static _CCCL_CONSTINIT ::cuda::std::atomic<terminate_handler>
    __cccl_terminate_handler{&__cccl_terminate};

_CCCL_API inline  terminate_handler set_terminate(terminate_handler __func) noexcept
{
  return __cccl_terminate_handler.exchange(__func);
}
_CCCL_API inline  terminate_handler get_terminate() noexcept
{
  return __cccl_terminate_handler.load(__func);
}

#endif

[[noreturn]] _CCCL_API inline void terminate() noexcept
{
  __cccl_terminate();
}

_CCCL_END_NAMESPACE_CUDA_STD_NOVERSION

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_TERMINATE_H
