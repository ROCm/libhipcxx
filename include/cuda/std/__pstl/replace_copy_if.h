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

#ifndef _CUDA_STD___PSTL_REPLACE_COPY_IF_H
#define _CUDA_STD___PSTL_REPLACE_COPY_IF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/replace_copy_if.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#  include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA() || _CCCL_HIP_COMPILATION() // HIP: hipCUB-backed backend
#    include <cuda/std/__pstl/cuda/transform.h>
#  endif // _CCCL_HAS_BACKEND_CUDA() || _CCCL_HIP_COMPILATION()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD
template <class _UnaryPred, class _Tp>
struct __replace_copy_if_select
{
  _UnaryPred __pred_;
  _Tp __new_value_;

  _CCCL_HOST_API constexpr __replace_copy_if_select(_UnaryPred __pred, const _Tp& __new_value) noexcept(
    is_nothrow_move_constructible_v<_UnaryPred> && is_nothrow_copy_constructible_v<_Tp>)
      : __pred_(__pred)
      , __new_value_(__new_value)
  {}

  template <class _Up>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp operator()(const _Up& __val) const
    noexcept(is_nothrow_invocable_v<const _UnaryPred&, const _Up&> && is_nothrow_copy_constructible_v<_Tp>)
  {
    return ::cuda::std::invoke(__pred_, __val) ? __new_value_ : static_cast<_Tp>(__val);
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(
  class _Policy, class _InputIterator, class _OutputIterator, class _UnaryPred, class _Tp = iter_value_t<_InputIterator>)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND __has_forward_traversal<_OutputIterator> _CCCL_AND
                 is_execution_policy_v<_Policy>)
_CCCL_HOST_API _OutputIterator replace_copy_if(
  [[maybe_unused]] const _Policy& __policy,
  _InputIterator __first,
  _InputIterator __last,
  _OutputIterator __result,
  _UnaryPred __pred,
  const _Tp& __new_value)
{
  static_assert(indirect_unary_predicate<_UnaryPred, _InputIterator>,
                "cuda::std::replace_copy_if: UnaryPred must satisfy indirect_unary_predicate<InputIterator>");

  if (__first == __last)
  {
    return __result;
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::replace_copy_if");
    return __dispatch(
      __policy,
      ::cuda::std::move(__first),
      ::cuda::std::move(__last),
      ::cuda::std::move(__result),
      __replace_copy_if_select{::cuda::std::move(__pred), __new_value});
  }
  else
  {
    static_assert(__always_false_v<_Policy>,
                  "Parallel cuda::std::replace_copy_if requires at least one selected backend");
    return ::cuda::std::replace_copy_if(
      ::cuda::std::move(__first),
      ::cuda::std::move(__last),
      ::cuda::std::move(__result),
      ::cuda::std::move(__pred),
      __new_value);
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_REPLACE_COPY_IF_H
