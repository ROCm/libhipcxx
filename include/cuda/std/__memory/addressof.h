// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _CUDA_STD___MEMORY_ADDRESSOF_H
#define _CUDA_STD___MEMORY_ADDRESSOF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE(HIP/AMD): hipRTC is clang-based but has no host C++ standard library, so
// `::std::addressof` is unavailable (the host <memory> include resolves to an
// empty stub). Exclude it from the builtin branch so it uses the
// __builtin_addressof-based manual implementation below instead.
#if (_CCCL_COMPILER(CLANG, >=, 15) || _CCCL_COMPILER(GCC, >=, 12)) && !defined(_CCCL_COMPILER_HIPRTC)
#  define _CCCL_HAS_BUILTIN_STD_ADDRESSOF() 1
#else // ^^^ has builtin std::addressof ^^^ / vvv no builtin std::addressof vvv
#  define _CCCL_HAS_BUILTIN_STD_ADDRESSOF() 0
#endif // ^^^ no builtin std::addressof ^^^

// nvcc warns about host only std::addressof being used in device code
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_HAS_BUILTIN_STD_ADDRESSOF
#  define _CCCL_HAS_BUILTIN_STD_ADDRESSOF() 0
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

// include minimal std:: headers
#if _CCCL_HAS_BUILTIN_STD_ADDRESSOF()
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX) && __has_include(<bits/move.h>)
#    include <bits/move.h>
#  elif _CCCL_HOST_STD_LIB(LIBCXX) && __has_include(<__memory/addressof.h>)
#    include <__memory/addressof.h>
#  else
#    include <cuda/std/__host_stdlib/memory>
#  endif
#endif // _CCCL_HAS_BUILTIN_STD_ADDRESSOF()

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4312) // warning C4312: 'type cast': conversion from '_Tp' to '_Tp *' of greater size

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_BUILTIN_STD_ADDRESSOF()

// The compiler treats ::std::addressof as a builtin function so it does not need to be
// instantiated and will be compiled away even at -O0.
using ::std::addressof;

#elif defined(_CCCL_BUILTIN_ADDRESSOF)

template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_NO_CFI constexpr _Tp* addressof(_Tp& __x) noexcept
{
  return _CCCL_BUILTIN_ADDRESSOF(__x);
}

template <class _Tp>
_Tp* addressof(const _Tp&&) noexcept = delete;

#else

template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_NO_CFI _Tp* addressof(_Tp& __x) noexcept
{
  return reinterpret_cast<_Tp*>(const_cast<char*>(&reinterpret_cast<const volatile char&>(__x)));
}

template <class _Tp>
_Tp* addressof(const _Tp&&) noexcept = delete;

#endif // defined(_CCCL_BUILTIN_ADDRESSOF)

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_ADDRESSOF_H
