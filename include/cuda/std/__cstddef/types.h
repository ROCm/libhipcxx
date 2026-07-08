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

#ifndef _CUDA_STD___CSTDDEF_TYPES_H
#define _CUDA_STD___CSTDDEF_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE(HIP/AMD): hipRTC has no host standard library and must take the freestanding
// path, but compiler.h does not fold HIPRTC into _CCCL_FREESTANDING()/_CCCL_HOSTED()
// (see __cccl/compiler.h freestanding gate), so mirror it explicitly here just as the
// pre-upgrade code did with !_CCCL_COMPILER(NVRTC) && !_CCCL_COMPILER_HIPRTC.
#if _CCCL_HOSTED()
#  include <cstddef>
#else // ^^^ hosted (non-HIPRTC) ^^^ / vvv freestanding or HIPRTC vvv
#  if !defined(offsetof)
// NOTE(HIP/AMD): C++ does not allow accessing a member through a null pointer in a constant expression.
// The following is a true constant expression and does not dereference a null pointer.
#  if defined(_CCCL_COMPILER_HIPRTC)
#    define offsetof(type, member) __builtin_offsetof(type, member)
#else
#    define offsetof(type, member) (::size_t) ((char*) &(((type*) 0)->member) - (char*) 0)
#endif
#  endif // !offsetof
#endif // freestanding or HIPRTC

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// NOTE(HIP/AMD): host <cstddef> is not included under hipRTC, so ::max_align_t
// is not in the global namespace; define it like NVRTC. HIPRTC is not folded into
// _CCCL_FREESTANDING() by compiler.h, so mirror it explicitly here.
#if _CCCL_FREESTANDING() || defined(_CCCL_COMPILER_HIPRTC)
using max_align_t = long double;
#else // ^^^ freestanding or HIPRTC ^^^ / vvv hosted vvv
// Re-use the compiler's <stddef.h> max_align_t where possible.
using ::max_align_t;
#endif // hosted

using nullptr_t = decltype(nullptr);
#if defined(_CCCL_COMPILER_HIPRTC)
// NOTE(HIP/AMD): host <cstddef> is not included under hipRTC and, unlike NVRTC's
// nvcc, the driver does not predefine ::ptrdiff_t/::size_t. Define them from the
// compiler builtins so cuda::std has them without pulling host headers.
using ptrdiff_t = __PTRDIFF_TYPE__;
using size_t    = __SIZE_TYPE__;
#else // ^^^ _CCCL_COMPILER_HIPRTC ^^^ / vvv !_CCCL_COMPILER_HIPRTC vvv
using ::ptrdiff_t;
using ::size_t;
#endif // _CCCL_COMPILER_HIPRTC

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CSTDDEF_TYPES_H
