// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#ifndef _CUDA___BARRIER_BARRIER_HIP_THREAD_SCOPE_H
#define _CUDA___BARRIER_BARRIER_HIP_THREAD_SCOPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier_hip_block_scope.h>
#include <cuda/__fwd/barrier.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// AMD HIP specialization for thread_scope_thread barriers.
//
// On AMD, thread_scope_thread is an alias to thread_scope_block semantics
// (single-thread barriers are not a distinct hardware primitive).
// This matches CUDA's behavior where thread_scope_thread barriers
// inherit from barrier<thread_scope_block>.

template <>
class barrier<thread_scope_thread, _CUDA_VSTD::__empty_completion>
  : private barrier<thread_scope_block, _CUDA_VSTD::__empty_completion>
{
  using __base = barrier<thread_scope_block, _CUDA_VSTD::__empty_completion>;

public:
  using __base::__base;

  _LIBCUDACXX_HIDE_FROM_ABI friend void init(barrier* __b, _CUDA_VSTD::ptrdiff_t __expected)
  {
    init(static_cast<__base*>(__b), __expected);
  }

  using __base::arrival_token;
  using __base::arrive;
  using __base::arrive_and_drop;
  using __base::arrive_and_wait;
  using __base::max;
  using __base::wait;
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_BARRIER_HIP_THREAD_SCOPE_H
