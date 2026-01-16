//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _CUDA___STREAM_GET_STREAM_H
#define _CUDA___STREAM_GET_STREAM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#  include <cuda/__fwd/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__concepts/convertible_to.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__type_traits/is_convertible.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

class stream_ref;

template <class _Tp>
_CCCL_CONCEPT __convertible_to_stream_ref = ::cuda::std::convertible_to<_Tp, ::cuda::stream_ref>;

template <class _Tp>
_CCCL_CONCEPT __has_member_stream = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __t)(
  requires(!__convertible_to_stream_ref<_Tp>), //
  requires(__convertible_to_stream_ref<decltype(__t.stream())>));

template <class _Tp>
_CCCL_CONCEPT __has_member_get_stream = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __t)(
  requires(!__convertible_to_stream_ref<_Tp>), //
  requires(__convertible_to_stream_ref<decltype(__t.get_stream())>));

template <class _Env>
_CCCL_CONCEPT __has_query_get_stream = _CCCL_REQUIRES_EXPR((_Env), const _Env& __env, const get_stream_t& __cpo)(
  requires(!__convertible_to_stream_ref<_Env>),
  requires(!__has_member_stream<_Env>),
  requires(__convertible_to_stream_ref<decltype(__env.query(__cpo))>));

//! @brief `get_stream` is a customization point object that queries a type `T` for an associated stream
struct get_stream_t
{
  [[nodiscard]] _CCCL_API constexpr ::cuda::stream_ref operator()(::cudaStream_t __stream) const noexcept
  {
    return ::cuda::stream_ref{__stream};
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__convertible_to_stream_ref<_Tp>)
  [[nodiscard]] _CCCL_API constexpr ::cuda::stream_ref operator()(const _Tp& __t) const
    noexcept(noexcept(static_cast<::cuda::stream_ref>(__t)))
  {
    return static_cast<::cuda::stream_ref>(__t);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__has_member_stream<_Tp>)
  [[nodiscard]] _CCCL_API constexpr ::cuda::stream_ref operator()(const _Tp& __t) const noexcept(noexcept(__t.stream()))
  {
    return __t.stream();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__has_member_get_stream<_Tp>)
  [[nodiscard]] _CCCL_API constexpr ::cuda::stream_ref operator()(const _Tp& __t) const
    noexcept(noexcept(__t.get_stream()))
  {
    return __t.get_stream();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__has_query_get_stream<_Env>)
  [[nodiscard]] _CCCL_API constexpr ::cuda::stream_ref operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)), "");
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(::cuda::std::execution::forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto get_stream = get_stream_t{};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#endif // _CUDA___STREAM_GET_STREAM_H
