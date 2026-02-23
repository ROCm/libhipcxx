//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _CUDA_STD___PSTL_CUDA_GENERATE_H
#define _CUDA_STD___PSTL_CUDA_GENERATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE(HIP/AMD): also enabled on HIP, where ::cub::DeviceTransform::Generate is
// provided by the curated namespace in <cuda/std/__pstl/cuda/__hipcub.h>.
#if _CCCL_HAS_BACKEND_CUDA() || _CCCL_HIP_COMPILATION()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wshadow")
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-local-typedef")

#  if _CCCL_HIP_COMPILATION()
#    include <cuda/std/__pstl/cuda/__hipcub.h> // curated ::cub (DeviceTransform::Generate)
#  else
#    include <cub/device/device_transform.cuh>
#  endif

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/generate_n.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/tuple>

#  if !_CCCL_HIP_COMPILATION() // on HIP cuda* comes from <libhipcxx/__amd/cuda_runtime.h>
#    include <cuda_runtime.h>
#  endif

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__generate_n, __execution_backend::__cuda>
{
  template <class _Policy, class _OutputIterator, class _UnaryOp>
  [[nodiscard]] _CCCL_HOST_API static _OutputIterator
  __par_impl(const _Policy& __policy, _OutputIterator __result, const int64_t __count, _UnaryOp __func)
  {
    auto __stream    = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);
    const auto __ret = __result + __count;

    // We pass the policy as an environment to device_transform
    _CCCL_TRY_CUDA_API(
      ::cub::DeviceTransform::Generate,
      "__pstl_cuda_generate: call to cub device_transform::Generate failed",
      ::cuda::std::move(__result),
      __count,
      ::cuda::std::move(__func),
      __stream);

    __stream.sync();
    return __ret;
  }

  _CCCL_TEMPLATE(class _Policy, class _OutputIterator, class _Size, class _UnaryOp)
  _CCCL_REQUIRES(__has_forward_traversal<_OutputIterator>)
  [[nodiscard]] _CCCL_HOST_API _OutputIterator
  operator()([[maybe_unused]] const _Policy& __policy, _OutputIterator __result, _Size __count, _UnaryOp __func) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      try
      {
        return __par_impl(__policy, ::cuda::std::move(__result), __count, ::cuda::std::move(__func));
      }
      catch (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == cudaErrorMemoryAllocation)
        {
          _CCCL_THROW(::std::bad_alloc);
        }
        else
        {
          throw __err;
        }
      }
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "__pstl_cuda_generate: CUDA backend of cuda::std::generate requires at least random access "
                    "iterators");
      return ::cuda::std::generate_n(::cuda::std::move(__result), __count, ::cuda::std::move(__func));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA() || _CCCL_HIP_COMPILATION()

#endif // _CUDA_STD___PSTL_CUDA_GENERATE_H
