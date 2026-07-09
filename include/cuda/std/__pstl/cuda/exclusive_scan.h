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

#ifndef _CUDA_STD___PSTL_CUDA_EXCLUSIVE_SCAN_H
#define _CUDA_STD___PSTL_CUDA_EXCLUSIVE_SCAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE(HIP/AMD): also enabled on HIP, where ::cub::DeviceScan resolves to a
// hipCUB-backed shim via <cuda/std/__pstl/cuda/__hipcub.h>.
#if _CCCL_HAS_BACKEND_CUDA() || _CCCL_HIP_COMPILATION()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wshadow")
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-local-typedef")
_CCCL_DIAG_SUPPRESS_CLANG("-Wignored-attributes")
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

#  if _CCCL_HIP_COMPILATION()
#    include <cuda/std/__pstl/cuda/__hipcub.h> // hipCUB device primitives + curated ::cub::DeviceScan
#  else
#    include <cub/device/device_scan.cuh>
#  endif

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__iterator/tabulate_output_iterator.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__numeric/exclusive_scan.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__exclusive_scan, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API static _OutputIterator __par_impl(
    const _Policy& __policy,
    _InputIterator __first,
    iter_difference_t<_InputIterator> __count,
    _OutputIterator __result,
    _BinaryOp __binary_op,
    _Tp __init)
  {
    // We pass the policy as an environment to DeviceScan
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceScan::ExclusiveScan,
      "__pstl_cuda_exclusive_scan: kernel launch of cub::DeviceScan::ExclusiveScan failed",
      ::cuda::std::move(__first),
      __result,
      ::cuda::std::move(__binary_op),
      __init,
      __count,
      __policy);

    // Get the stream for synchronization after the algorithm is run
    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, __policy);
    __stream.sync();

    return __result + iter_difference_t<_OutputIterator>(__count);
  }

  template <class _Policy, class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _OutputIterator __result,
    _Tp __init,
    _BinaryOp __binary_op) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      _CCCL_TRY
      {
        const auto __count = ::cuda::std::distance(__first, __last);
        return __par_impl(
          __policy,
          ::cuda::std::move(__first),
          __count,
          ::cuda::std::move(__result),
          ::cuda::std::move(__binary_op),
          __init);
      }
      _CCCL_CATCH (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == cudaErrorMemoryAllocation)
        {
          _CCCL_THROW(::std::bad_alloc);
        }
        else
        {
          _CCCL_RETHROW;
        }
      }
      _CCCL_CATCH_FALLTHROUGH
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "__pstl_dispatch: CUDA backend of cuda::std::exclusive_scan requires at least random access "
                    "iterators");
      return ::cuda::std::exclusive_scan(
        ::cuda::std::move(__first),
        ::cuda::std::move(__last),
        ::cuda::std::move(__result),
        ::cuda::std::move(__binary_op),
        __init);
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA() || _CCCL_HIP_COMPILATION()

#endif // _CUDA_STD___PSTL_CUDA_EXCLUSIVE_SCAN_H
