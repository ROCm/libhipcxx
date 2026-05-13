//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

#ifndef _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H
#define _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// <<<<<<< OLD CODE from cfac2f6346 (521ff4297e) - COMMENTED OUT
// #if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
//
// // NOTE(HIP/AMD): the CUDA vector type names (char1, uint3, float4, ...) are
// // provided on HIP by <hip/hip_vector_types.h> as HIP_vector_type<T,N> aliases.
// #  if _CCCL_HIP_COMPILATION()
// #    include <hip/hip_vector_types.h>
// #  elif !_CCCL_CUDA_COMPILATION()
// #    include <vector_types.h>
// #  endif // vector type definitions
// =======
#if _CCCL_HAS_CTK()
#  include <cuda/__type_traits/scalar_type.h>
#  include <cuda/__type_traits/vector_size.h>
#  include <cuda/std/__floating_point/traits.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/void_t.h>
// >>>>>>> END NEW CODE (521ff4297e)

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// is_vector_type

template <class _Tp>
inline constexpr bool is_vector_type_v = (vector_size_v<_Tp> != 0);

template <class _Tp>
using is_vector_type = ::cuda::std::bool_constant<is_vector_type_v<_Tp>>;

// is_extended_fp_vector_type

template <class _Tp, class = void>
inline constexpr bool is_extended_fp_vector_type_v = false;
template <class _Tp>
inline constexpr bool is_extended_fp_vector_type_v<_Tp, ::cuda::std::void_t<typename scalar_type<_Tp>::type>> =
  ::cuda::std::__is_ext_nv_fp_v<scalar_type_t<_Tp>>;

template <class _Tp>
using is_extended_fp_vector_type = ::cuda::std::bool_constant<is_extended_fp_vector_type_v<_Tp>>;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
#endif // _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H
