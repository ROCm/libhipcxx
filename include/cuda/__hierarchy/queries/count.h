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

#ifndef _CUDA___HIERARCHY_QUERIES_COUNT_H
#define _CUDA___HIERARCHY_QUERIES_COUNT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/std/__cstddef/types.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// native hierarchy queries

#  if _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()

// cudafe++ makes the queries (that are device only) return void when compiling for host, which causes host compilers
// to warn about applying [[nodiscard]] to a function that returns void.
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_NVHPC(nodiscard_doesnt_apply)
#    if _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_CLANG("-Wignored-attributes")
#    endif // _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)

template <class _Unit, class _Level>
struct __count_query_native
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    const auto __exts = __extents_query_native<_Unit, _Level>::template __call<_Tp>();

    _Tp __ret = 1;
    for (::cuda::std::size_t __i = 0; __i < __exts.rank(); ++__i)
    {
      __ret *= __exts.extent(__i);
    }
    return __ret;
  }
};

template <>
struct __count_query_native<block_level, cluster_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    unsigned __count = 1;
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__count = ::__clusterSizeInBlocks();))
    return static_cast<_Tp>(__count);
  }
};

template <>
struct __count_query_native<block_level, grid_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    return static_cast<_Tp>(static_cast<_Tp>(gridDim.x) * gridDim.y * gridDim.z);
  }
};

_CCCL_DIAG_POP
#  endif // _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()

// hierarchy queries

template <class _Unit, class _Level>
struct __count_query
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_API static constexpr _Tp __call(const _Hierarchy& __hier) noexcept
  {
    const auto __exts = __extents_query<_Unit, _Level>::template __call<_Tp>(__hier);

    _Tp __ret = 1;
    for (::cuda::std::size_t __i = 0; __i < __exts.rank(); ++__i)
    {
      __ret *= __exts.extent(__i);
    }
    return __ret;
  }
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#endif // _CUDA___HIERARCHY_QUERIES_COUNT_H
