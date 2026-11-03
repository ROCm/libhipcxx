//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

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

// Legacy (libhipcxx <= 3.1) macros that upstream CCCL removed or renamed, kept
// for consumers that have not migrated yet (rocThrust and hipCUB in hipCCL).
// They detect libhipcxx via _CUDA_VSTD and use these macros in their own
// headers. libhipcxx itself must not use anything from this file.
//
// The matching legacy internal symbols (__identity, __invoke_of,
// __is_cpp17_*_iterator, __is_tuple_of_iterator_references,
// __throw_runtime_error) and the <cuda/std/detail/libcxx/include/*> forwarding
// headers are guarded by the same opt-out macro.
//
// Define LIBHIPCXX_NO_LEGACY_COMPAT to disable all of it.
//
// The compat mode is deprecated: every legacy name warns where a consumer uses
// it. Define LIBHIPCXX_IGNORE_DEPRECATED_LEGACY_COMPAT (or
// CCCL_IGNORE_DEPRECATED_API) to silence the warnings. Macros can only be
// marked deprecated with clang.

#ifndef _AMD_LEGACY_COMPAT_H
#define _AMD_LEGACY_COMPAT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/deprecated.h>
#include <cuda/std/__cccl/exceptions.h>

#ifndef LIBHIPCXX_NO_LEGACY_COMPAT

#  define _LIBHIPCXX_LEGACY_COMPAT_MSG                                                                 \
    "libhipcxx legacy compat mode (libhipcxx <= 3.1 API) is deprecated and will be removed soon. Migrate " \
    "to the libhipcxx >= 3.2 API, or define LIBHIPCXX_IGNORE_DEPRECATED_LEGACY_COMPAT to silence this "    \
    "warning."

#  if defined(LIBHIPCXX_IGNORE_DEPRECATED_LEGACY_COMPAT) || defined(CCCL_IGNORE_DEPRECATED_API)
#    define _LIBHIPCXX_LEGACY_DEPRECATED
#    define _LIBHIPCXX_LEGACY_DEPRECATE_MACROS() 0
#  else // ^^^ silenced ^^^ / vvv warn vvv
#    define _LIBHIPCXX_LEGACY_DEPRECATED         CCCL_DEPRECATED_BECAUSE(_LIBHIPCXX_LEGACY_COMPAT_MSG)
#    define _LIBHIPCXX_LEGACY_DEPRECATE_MACROS() _CCCL_COMPILER(CLANG)
#  endif // ^^^ warn ^^^

// Namespaces, removed in CCCL #5538 / #6548. The definitions are the 3.0 ones
// and deliberately skip _CCCL_PROLOGUE_INCLUDE_CHECK(), as they are expanded
// in consumer headers.
// clang-format off
#  define _LIBCUDACXX_BEGIN_NAMESPACE_STD  namespace cuda { namespace std { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_STD    } } } namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA namespace cuda { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA   } } namespace hip = cuda;
#  define _CUDA_VSTD_NOVERSION             ::cuda::std
#  define _CUDA_VSTD                       ::cuda::std::_LIBCUDACXX_ABI_NAMESPACE
// clang-format on

// Attributes and helpers
#  ifndef _CCCL_NODISCARD
#    define _CCCL_NODISCARD [[nodiscard]]
#  endif
#  ifndef _CCCL_NODISCARD_FRIEND
#    define _CCCL_NODISCARD_FRIEND [[nodiscard]] friend
#  endif
#  ifndef _CCCL_NORETURN
#    define _CCCL_NORETURN [[noreturn]]
#  endif
#  ifndef _CCCL_ALIAS_ATTRIBUTE
#    define _CCCL_ALIAS_ATTRIBUTE(...) __VA_ARGS__
#  endif
#  ifndef _CCCL_HAS_INCLUDE
#    define _CCCL_HAS_INCLUDE(_X) __has_include(_X)
#  endif
#  ifndef _CCCL_TRAIT
#    define _CCCL_TRAIT(__TRAIT, ...) __TRAIT##_v<__VA_ARGS__>
#  endif
// In libhipcxx 3.0 this was also true for hipcc and hiprtc.
#  ifndef _CCCL_HAS_CUDA_COMPILER
#    define _CCCL_HAS_CUDA_COMPILER() (_CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION())
#  endif
#  if !_CCCL_HAS_EXCEPTIONS() && !defined(_CCCL_NO_EXCEPTIONS)
#    define _CCCL_NO_EXCEPTIONS
#  endif

#  if _LIBHIPCXX_LEGACY_DEPRECATE_MACROS()
#    pragma clang deprecated(_LIBCUDACXX_BEGIN_NAMESPACE_STD, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_LIBCUDACXX_END_NAMESPACE_STD, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_LIBCUDACXX_BEGIN_NAMESPACE_CUDA, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_LIBCUDACXX_END_NAMESPACE_CUDA, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_CUDA_VSTD_NOVERSION, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_CUDA_VSTD, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_CCCL_NODISCARD, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_CCCL_NODISCARD_FRIEND, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_CCCL_NORETURN, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_CCCL_ALIAS_ATTRIBUTE, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_CCCL_HAS_INCLUDE, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_CCCL_TRAIT, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    pragma clang deprecated(_CCCL_HAS_CUDA_COMPILER, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    ifdef _CCCL_NO_EXCEPTIONS
#      pragma clang deprecated(_CCCL_NO_EXCEPTIONS, _LIBHIPCXX_LEGACY_COMPAT_MSG)
#    endif
#  endif // _LIBHIPCXX_LEGACY_DEPRECATE_MACROS()

#endif // !LIBHIPCXX_NO_LEGACY_COMPAT

#endif // _AMD_LEGACY_COMPAT_H
