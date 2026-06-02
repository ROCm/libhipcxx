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

#ifndef __CCCL_EXCEPTIONS_H
#define __CCCL_EXCEPTIONS_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(CCCL_DISABLE_EXCEPTIONS) // Escape hatch for users to manually disable exceptions
#  define _CCCL_HAS_EXCEPTIONS() 0
#elif _CCCL_COMPILER(NVRTC) || defined(_CCCL_COMPILER_HIPRTC) // NVRTC/HIPRTC have no exceptions
#  define _CCCL_HAS_EXCEPTIONS() 0
#elif _CCCL_COMPILER(MSVC) // MSVC needs special checks for `_HAS_EXCEPTIONS` and `_CPPUNWIND`
#  define _CCCL_HAS_EXCEPTIONS() ((_HAS_EXCEPTIONS != 0) && (_CPPUNWIND != 0)) // disabled with /EH
#elif defined(__HIP_PLATFORM_AMD__) // HIP host platform assumed to have exceptions enabled
#  define _CCCL_HAS_EXCEPTIONS() 1
#else // other compilers use `__EXCEPTIONS`
#  define _CCCL_HAS_EXCEPTIONS() (__EXCEPTIONS) // disabled with -fno-exceptions
#endif // has exceptions

#if _CCCL_HAS_EXCEPTIONS() && __cpp_constexpr_exceptions >= 202411L
#  define _CCCL_HAS_CONSTEXPR_EXCEPTIONS() 1
#else // ^^^ has constexpr exceptions ^^^ / vvv no constexpr exceptions vvv
#  define _CCCL_HAS_CONSTEXPR_EXCEPTIONS() 0
#endif // ^^^ no constexpr exceptions ^^^

#endif // __CCCL_EXCEPTIONS_H
