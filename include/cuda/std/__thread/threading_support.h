// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from 8ac92af3c4 (de9c251c78) - COMMENTED OUT
// // MIT License
// //
// // Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
// #ifndef _LIBCUDACXX___THREAD_THREADING_SUPPORT_H
// #define _LIBCUDACXX___THREAD_THREADING_SUPPORT_H
// =======
#ifndef _CUDA_STD___THREAD_THREADING_SUPPORT_H
#define _CUDA_STD___THREAD_THREADING_SUPPORT_H
// >>>>>>> END NEW CODE (de9c251c78)

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/chrono>

#if defined(_LIBCUDACXX_HAS_THREAD_API_EXTERNAL)
#  include <cuda/std/__thread/threading_support_external.h>
#endif // _LIBCUDACXX_HAS_THREAD_API_EXTERNAL

#if defined(_LIBCUDACXX_HAS_THREAD_API_CUDA)
#  include <cuda/std/__thread/threading_support_cuda.h>
#elif defined(_LIBCUDACXX_HAS_THREAD_API_PTHREAD)
#  include <cuda/std/__thread/threading_support_pthread.h>
#elif defined(_LIBCUDACXX_HAS_THREAD_API_WIN32)
#  include <cuda/std/__thread/threading_support_win32.h>
#else // ^^^ _LIBCUDACXX_HAS_THREAD_API_WIN32 ^^^ / vvv Unknown Thread API vvv
#  error "Unknown Thread API"
#endif // Unknown Thread API

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#define _LIBCUDACXX_POLLING_COUNT 16

#if _CCCL_ARCH(ARM64) && _CCCL_OS(LINUX)
#  define __LIBCUDACXX_ASM_THREAD_YIELD (asm volatile("yield" :: :);)
#elif _CCCL_ARCH(X86_64) && _CCCL_OS(LINUX)
#  define __LIBCUDACXX_ASM_THREAD_YIELD (asm volatile("pause" :: :);)
#else // ^^^  _CCCL_ARCH(X86_64) ^^^ / vvv ! _CCCL_ARCH(X86_64) vvv
#  define __LIBCUDACXX_ASM_THREAD_YIELD (;)
#endif // ! _CCCL_ARCH(X86_64)

_CCCL_API inline void __cccl_thread_yield_processor()
{
  NV_IF_TARGET(NV_IS_HOST, __LIBCUDACXX_ASM_THREAD_YIELD)
}

template <class _Fn>
_CCCL_API inline bool __cccl_thread_poll_with_backoff(
  _Fn&& __f, ::cuda::std::chrono::nanoseconds __max = ::cuda::std::chrono::nanoseconds::zero())
{
  ::cuda::std::chrono::high_resolution_clock::time_point const __start =
    ::cuda::std::chrono::high_resolution_clock::now();
  for (int __count = 0;;)
  {
    if (__f())
    {
      return true;
    }
    if (__count < _LIBCUDACXX_POLLING_COUNT)
    {
      if (__count > (_LIBCUDACXX_POLLING_COUNT >> 1))
      {
        ::cuda::std::__cccl_thread_yield_processor();
      }
      __count += 1;
      continue;
    }
    ::cuda::std::chrono::high_resolution_clock::duration const __elapsed =
      ::cuda::std::chrono::high_resolution_clock::now() - __start;
    if (__max != ::cuda::std::chrono::nanoseconds::zero() && __max < __elapsed)
    {
      return false;
    }
    ::cuda::std::chrono::nanoseconds const __step = __elapsed / 4;
    if (__step >= ::cuda::std::chrono::milliseconds(1))
    {
      ::cuda::std::__cccl_thread_sleep_for(::cuda::std::chrono::milliseconds(1));
    }
    else if (__step >= ::cuda::std::chrono::microseconds(10))
    {
      ::cuda::std::__cccl_thread_sleep_for(__step);
    }
    else
    {
      ::cuda::std::__cccl_thread_yield();
    }
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___THREAD_THREADING_SUPPORT_H
