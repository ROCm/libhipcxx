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

#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

// NOTE(HIP/AMD): the CUDA vector type names (char1, uint3, float4, ...) are
// provided on HIP by <hip/hip_vector_types.h> as HIP_vector_type<T,N> aliases.
#  if _CCCL_HIP_COMPILATION()
#    include <hip/hip_vector_types.h>
#  elif !_CCCL_CUDA_COMPILATION()
#    include <vector_types.h>
#  endif // vector type definitions
#  if _CCCL_HAS_CTK()
#    include <cuda/__type_traits/scalar_type.h>
#    include <cuda/__type_traits/vector_size.h>
#    include <cuda/std/__floating_point/traits.h>
#    include <cuda/std/__type_traits/integral_constant.h>
#    include <cuda/std/__type_traits/void_t.h>
#  else // _CCCL_HIP_COMPILATION() && !_CCCL_HAS_CTK()
#    include <cuda/std/__type_traits/integral_constant.h>
#  endif // _CCCL_HAS_CTK()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#  if _CCCL_HAS_CTK()

// is_vector_type (CUDA/CTK path: derived from vector_size_v)

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

#  else // _CCCL_HIP_COMPILATION() && !_CCCL_HAS_CTK()

// NOTE(HIP/AMD): on HIP without CTK, vector_size_v and scalar_type are not
// available (they gate on _CCCL_HAS_CTK()). Provide direct bool specializations
// matching the AMD integration approach (g019 / phase-3 adaptation).
_CCCL_SUPPRESS_DEPRECATED_PUSH

// is_vector_type (HIP fallback: explicit specializations)

template <class _Tp>
inline constexpr bool is_vector_type_v = false;
template <class _Tp>
inline constexpr bool is_vector_type_v<const _Tp> = is_vector_type_v<_Tp>;
template <class _Tp>
inline constexpr bool is_vector_type_v<volatile _Tp> = is_vector_type_v<_Tp>;
template <class _Tp>
inline constexpr bool is_vector_type_v<const volatile _Tp> = is_vector_type_v<_Tp>;

template <>
inline constexpr bool is_vector_type_v<::char1> = true;
template <>
inline constexpr bool is_vector_type_v<::char2> = true;
template <>
inline constexpr bool is_vector_type_v<::char3> = true;
template <>
inline constexpr bool is_vector_type_v<::char4> = true;

template <>
inline constexpr bool is_vector_type_v<::uchar1> = true;
template <>
inline constexpr bool is_vector_type_v<::uchar2> = true;
template <>
inline constexpr bool is_vector_type_v<::uchar3> = true;
template <>
inline constexpr bool is_vector_type_v<::uchar4> = true;

template <>
inline constexpr bool is_vector_type_v<::short1> = true;
template <>
inline constexpr bool is_vector_type_v<::short2> = true;
template <>
inline constexpr bool is_vector_type_v<::short3> = true;
template <>
inline constexpr bool is_vector_type_v<::short4> = true;

template <>
inline constexpr bool is_vector_type_v<::ushort1> = true;
template <>
inline constexpr bool is_vector_type_v<::ushort2> = true;
template <>
inline constexpr bool is_vector_type_v<::ushort3> = true;
template <>
inline constexpr bool is_vector_type_v<::ushort4> = true;

template <>
inline constexpr bool is_vector_type_v<::int1> = true;
template <>
inline constexpr bool is_vector_type_v<::int2> = true;
template <>
inline constexpr bool is_vector_type_v<::int3> = true;
template <>
inline constexpr bool is_vector_type_v<::int4> = true;

template <>
inline constexpr bool is_vector_type_v<::uint1> = true;
template <>
inline constexpr bool is_vector_type_v<::uint2> = true;
template <>
inline constexpr bool is_vector_type_v<::uint3> = true;
template <>
inline constexpr bool is_vector_type_v<::uint4> = true;

template <>
inline constexpr bool is_vector_type_v<::long1> = true;
template <>
inline constexpr bool is_vector_type_v<::long2> = true;
template <>
inline constexpr bool is_vector_type_v<::long3> = true;
template <>
inline constexpr bool is_vector_type_v<::long4> = true;

template <>
inline constexpr bool is_vector_type_v<::ulong1> = true;
template <>
inline constexpr bool is_vector_type_v<::ulong2> = true;
template <>
inline constexpr bool is_vector_type_v<::ulong3> = true;
template <>
inline constexpr bool is_vector_type_v<::ulong4> = true;

template <>
inline constexpr bool is_vector_type_v<::longlong1> = true;
template <>
inline constexpr bool is_vector_type_v<::longlong2> = true;
template <>
inline constexpr bool is_vector_type_v<::longlong3> = true;
template <>
inline constexpr bool is_vector_type_v<::longlong4> = true;

template <>
inline constexpr bool is_vector_type_v<::ulonglong1> = true;
template <>
inline constexpr bool is_vector_type_v<::ulonglong2> = true;
template <>
inline constexpr bool is_vector_type_v<::ulonglong3> = true;
template <>
inline constexpr bool is_vector_type_v<::ulonglong4> = true;

template <>
inline constexpr bool is_vector_type_v<::float1> = true;
template <>
inline constexpr bool is_vector_type_v<::float2> = true;
template <>
inline constexpr bool is_vector_type_v<::float3> = true;
template <>
inline constexpr bool is_vector_type_v<::float4> = true;

template <>
inline constexpr bool is_vector_type_v<::double1> = true;
template <>
inline constexpr bool is_vector_type_v<::double2> = true;
template <>
inline constexpr bool is_vector_type_v<::double3> = true;
template <>
inline constexpr bool is_vector_type_v<::double4> = true;

// dim3 is a vector type on the CUDA path (vector_size_v<dim3> == 3); mirror that
// here for parity (matches the amd-integration-base adaptation).
template <>
inline constexpr bool is_vector_type_v<::dim3> = true;

// NOTE(HIP/AMD): __half2 and __nv_bfloat162 are vector types on both CUDA and HIP
// (vector_size_v<__half2> == 2 on the CTK path). Mirror that here for parity.
#    if _CCCL_HAS_NVFP16()
template <>
inline constexpr bool is_vector_type_v<::__half2> = true;
#    endif // _CCCL_HAS_NVFP16()

#    if _CCCL_HAS_NVBF16()
template <>
inline constexpr bool is_vector_type_v<::__nv_bfloat162> = true;
#    endif // _CCCL_HAS_NVBF16()

// NOTE(HIP/AMD): HIP ships __hip_fp8x2_e4m3/__hip_fp8x4_e4m3/__hip_fp8x2_e5m2/
// __hip_fp8x4_e5m2 in <hip/hip_fp8.h>; <libhipcxx/__amd/cuda_runtime.h> aliases
// them to the upstream __nv_fp8x* names. These are vector types analogous to
// __half2 / __nv_bfloat162 (vector_size_v<__nv_fp8x2_e4m3> == 2 on the CTK path).
#    if _CCCL_HAS_NVFP8()
template <>
inline constexpr bool is_vector_type_v<::__nv_fp8x2_e4m3> = true;
template <>
inline constexpr bool is_vector_type_v<::__nv_fp8x4_e4m3> = true;
template <>
inline constexpr bool is_vector_type_v<::__nv_fp8x2_e5m2> = true;
template <>
inline constexpr bool is_vector_type_v<::__nv_fp8x4_e5m2> = true;
#    endif // _CCCL_HAS_NVFP8()

template <class _Tp>
using is_vector_type = ::cuda::std::bool_constant<is_vector_type_v<_Tp>>;

// is_extended_fp_vector_type (HIP fallback: explicit specializations)

template <class _Tp>
inline constexpr bool is_extended_fp_vector_type_v = false;
// NOTE(HIP/AMD): strip cv qualifiers so const/volatile specializations work.
template <class _Tp>
inline constexpr bool is_extended_fp_vector_type_v<const _Tp> = is_extended_fp_vector_type_v<_Tp>;
template <class _Tp>
inline constexpr bool is_extended_fp_vector_type_v<volatile _Tp> = is_extended_fp_vector_type_v<_Tp>;
template <class _Tp>
inline constexpr bool is_extended_fp_vector_type_v<const volatile _Tp> = is_extended_fp_vector_type_v<_Tp>;

#    if _CCCL_HAS_NVFP16()
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__half2> = true;
#    endif // _CCCL_HAS_NVFP16()

#    if _CCCL_HAS_NVBF16()
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_bfloat162> = true;
#    endif // _CCCL_HAS_NVBF16()

// NOTE(HIP/AMD): HIP ships __hip_fp8x2_e4m3/__hip_fp8x4_e4m3/__hip_fp8x2_e5m2/
// __hip_fp8x4_e5m2 in <hip/hip_fp8.h>; <libhipcxx/__amd/cuda_runtime.h> aliases
// them to the upstream __nv_fp8x* names. Add is_extended_fp_vector_type_v
// specializations so that __data_type_to_dlpack<> can decompose them into their
// scalar element type via tuple_size_v / tuple_element_t (defined in
// cuda/std/__tuple_dir/vector_types.h, guarded by _CCCL_HAS_NVFP8()).
#    if _CCCL_HAS_NVFP8()
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x2_e4m3> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x4_e4m3> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x2_e5m2> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x4_e5m2> = true;
#    endif // _CCCL_HAS_NVFP8()

template <class _Tp>
using is_extended_fp_vector_type = ::cuda::std::bool_constant<is_extended_fp_vector_type_v<_Tp>>;

_CCCL_SUPPRESS_DEPRECATED_POP

#  endif // _CCCL_HAS_CTK() vs. HIP fallback

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
#endif // _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H
