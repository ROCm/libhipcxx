// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#ifndef _CUDA_STD___THREAD_THREADING_SUPPORT_WIN32_H
#define _CUDA_STD___THREAD_THREADING_SUPPORT_WIN32_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_CCCL_HAS_THREAD_API_WIN32)

#  include <cuda/std/__chrono/duration.h>

#  include <process.h>

// NOTE(HIP/AMD): <windows.h> defines the function-like macros min(a, b) and max(a, b) unless NOMINMAX is defined. The
// prologue below can only protect against macros that are already defined, so it cannot undo a leak from this include.
// Prevent windows.h from defining them in the first place and restore the includer's NOMINMAX state afterwards, so that
// including libhipcxx headers does not break user code such as std::min(...) or numeric_limits<T>::max().
#  pragma push_macro("NOMINMAX")
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif // !NOMINMAX
#  include <windows.h>
#  pragma pop_macro("NOMINMAX")

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Mutex
using __cccl_mutex_t = void*;
#  define _LIBCUDACXX_MUTEX_INITIALIZER 0

#  if _CCCL_ARCH(ARM64) || _CCCL_ARCH(X86_64)
using __cccl_recursive_mutex_t = void* [5];
#  else
#    error Unsupported architecture
#  endif

// Condition Variable
using __cccl_condvar_t = void*;
#  define _LIBCUDACXX_CONDVAR_INITIALIZER 0

// Semaphore
using __cccl_semaphore_t = void*;

// Execute Once
using __cccl_exec_once_flag = void*;
#  define _LIBCUDACXX_EXEC_ONCE_INITIALIZER 0

// Thread ID
using __cccl_thread_id = long;

// Thread
#  define _LIBCUDACXX_NULL_THREAD 0U

using __cccl_thread_t = void*;

// Thread Local Storage
using __cccl_tls_key = long;

#  define _LIBCUDACXX_TLS_DESTRUCTOR_CC __stdcall

_CCCL_API inline void __cccl_thread_yield()
{
  SwitchToThread();
}

_CCCL_API inline void __cccl_thread_sleep_for(chrono::nanoseconds __ns)
{
  using namespace chrono;
  // round-up to the nearest millisecond
  milliseconds __ms = duration_cast<milliseconds>(__ns + chrono::nanoseconds(999999));
  Sleep(static_cast<DWORD>(__ms.count()));
}

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_THREAD_API_WIN32

#endif // _CUDA_STD___THREAD_THREADING_SUPPORT_H
