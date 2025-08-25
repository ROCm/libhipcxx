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

// <<<<<<< OLD CODE from 8ac92af3c4 (de9c251c78) - COMMENTED OUT
// // Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// // Permission is hereby granted, free of charge, to any person obtaining a copy
// // of this software and associated documentation files (the "Software"), to deal
// // in the Software without restriction, including without limitation the rights
// // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // copies of the Software, and to permit persons to whom the Software is
// // furnished to do so, subject to the following conditions:
// // The above copyright notice and this permission notice shall be included in
// // all copies or substantial portions of the Software.
// // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// // THE SOFTWARE.
//
// #ifndef _LIBCUDACXX___EXCEPTION_TERMINATE_H
// #define _LIBCUDACXX___EXCEPTION_TERMINATE_H
// =======
#ifndef _CUDA_STD___EXCEPTION_TERMINATE_H
#define _CUDA_STD___EXCEPTION_TERMINATE_H
// >>>>>>> END NEW CODE (de9c251c78)

#include <cuda/std/detail/__config>
#include <amd/amd_utils.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdlib> // ::exit

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code

_CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION // purposefully not using versioning namespace

[[noreturn]] _CCCL_API inline void __cccl_terminate() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::exit(-1);), (libhipcxx::__trap();))
  _CCCL_UNREACHABLE();
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
  _CCCL_UNREACHABLE();
}

_CCCL_END_NAMESPACE_CUDA_STD_NOVERSION

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_TERMINATE_H
