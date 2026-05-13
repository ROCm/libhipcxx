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

#ifndef _CUDA__TYPE_TRAITS_VECTOR_TYPE_H
#define _CUDA__TYPE_TRAITS_VECTOR_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/is_same.h>

// <<<<<<< OLD CODE from cfac2f6346 (521ff4297e) - COMMENTED OUT
// // NOTE(HIP/AMD): the CUDA vector type names (char1, uint3, float4, ...) are
// // provided on HIP by <hip/hip_vector_types.h> as HIP_vector_type<T,N> aliases.
// #  if _CCCL_HIP_COMPILATION()
// #    include <hip/hip_vector_types.h>
// #  elif !_CCCL_CUDA_COMPILATION()
// #    include <vector_types.h>
// #  endif // vector type definitions
//
// =======
// >>>>>>> END NEW CODE (521ff4297e)
#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// vector_type

template <class _Tp, ::cuda::std::size_t _Size>
// <<<<<<< OLD CODE from cfac2f6346 (521ff4297e) - COMMENTED OUT
// [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __cccl_vector_type_t_impl() noexcept
// {
//   // NOTE(HIP/AMD): the element type of HIP's charN is plain `char` (not the
//   // `signed char` used by CUDA), matching the tuple-element specialization in
//   // <cuda/std/__tuple_dir/vector_types.h>. Map `char` -> charN on HIP so that
//   // vector_type_t and scalar_type_t stay consistent.
// #  if _CCCL_HIP_COMPILATION()
//   if constexpr (::cuda::std::is_same_v<_Tp, char>)
// #  else // ^^^ _CCCL_HIP_COMPILATION() ^^^ / vvv !_CCCL_HIP_COMPILATION() vvv
//   if constexpr (::cuda::std::is_same_v<_Tp, signed char>)
// #  endif // !_CCCL_HIP_COMPILATION()
//   {
//     if constexpr (_Size == 1)
//     {
//       return ::char1{};
//     }
//     else if constexpr (_Size == 2)
//     {
//       return ::char2{};
//     }
//     else if constexpr (_Size == 3)
//     {
//       return ::char3{};
//     }
//     else if constexpr (_Size == 4)
//     {
//       return ::char4{};
//     }
//     else
//     {
//       return;
//     }
//   }
//   else if constexpr (::cuda::std::is_same_v<_Tp, unsigned char>)
//   {
//     if constexpr (_Size == 1)
//     {
//       return ::uchar1{};
//     }
//     else if constexpr (_Size == 2)
//     {
//       return ::uchar2{};
//     }
//     else if constexpr (_Size == 3)
//     {
//       return ::uchar3{};
//     }
//     else if constexpr (_Size == 4)
//     {
//       return ::uchar4{};
//     }
//     else
//     {
//       return;
//     }
//   }
//   else if constexpr (::cuda::std::is_same_v<_Tp, short>)
//   {
//     if constexpr (_Size == 1)
//     {
//       return ::short1{};
//     }
//     else if constexpr (_Size == 2)
//     {
//       return ::short2{};
//     }
//     else if constexpr (_Size == 3)
//     {
//       return ::short3{};
//     }
//     else if constexpr (_Size == 4)
//     {
//       return ::short4{};
//     }
//     else
//     {
//       return;
//     }
//   }
//   else if constexpr (::cuda::std::is_same_v<_Tp, unsigned short>)
//   {
//     if constexpr (_Size == 1)
//     {
//       return ::ushort1{};
//     }
//     else if constexpr (_Size == 2)
//     {
//       return ::ushort2{};
//     }
//     else if constexpr (_Size == 3)
//     {
//       return ::ushort3{};
//     }
//     else if constexpr (_Size == 4)
//     {
//       return ::ushort4{};
//     }
//     else
//     {
//       return;
//     }
//   }
//   else if constexpr (::cuda::std::is_same_v<_Tp, int>)
//   {
//     if constexpr (_Size == 1)
//     {
//       return ::int1{};
//     }
//     else if constexpr (_Size == 2)
//     {
//       return ::int2{};
//     }
//     else if constexpr (_Size == 3)
//     {
//       return ::int3{};
//     }
//     else if constexpr (_Size == 4)
//     {
//       return ::int4{};
//     }
//     else
//     {
//       return;
//     }
//   }
//   else if constexpr (::cuda::std::is_same_v<_Tp, unsigned>)
//   {
//     if constexpr (_Size == 1)
//     {
//       return ::uint1{};
//     }
//     else if constexpr (_Size == 2)
//     {
//       return ::uint2{};
//     }
//     else if constexpr (_Size == 3)
//     {
//       return ::uint3{};
//     }
//     else if constexpr (_Size == 4)
//     {
//       return ::uint4{};
//     }
//     else
//     {
//       return;
//     }
//   }
//   else if constexpr (::cuda::std::is_same_v<_Tp, long>)
//   {
//     if constexpr (_Size == 1)
//     {
//       return ::long1{};
//     }
//     else if constexpr (_Size == 2)
//     {
//       return ::long2{};
//     }
//     else if constexpr (_Size == 3)
//     {
//       return ::long3{};
//     }
//     else if constexpr (_Size == 4)
//     {
// =======
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type
{
  using type = void;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<signed char, 1>
{
  using type = ::char1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<signed char, 2>
{
  using type = ::char2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<signed char, 3>
{
  using type = ::char3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<signed char, 4>
{
  using type = ::char4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned char, 1>
{
  using type = ::uchar1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned char, 2>
{
  using type = ::uchar2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned char, 3>
{
  using type = ::uchar3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned char, 4>
{
  using type = ::uchar4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<short, 1>
{
  using type = ::short1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<short, 2>
{
  using type = ::short2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<short, 3>
{
  using type = ::short3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<short, 4>
{
  using type = ::short4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned short, 1>
{
  using type = ::ushort1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned short, 2>
{
  using type = ::ushort2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned short, 3>
{
  using type = ::ushort3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned short, 4>
{
  using type = ::ushort4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<int, 1>
{
  using type = ::int1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<int, 2>
{
  using type = ::int2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<int, 3>
{
  using type = ::int3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<int, 4>
{
  using type = ::int4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned, 1>
{
  using type = ::uint1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned, 2>
{
  using type = ::uint2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned, 3>
{
  using type = ::uint3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned, 4>
{
  using type = ::uint4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long, 1>
{
  using type = ::long1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long, 2>
{
  using type = ::long2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long, 3>
{
  using type = ::long3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long, 4>
{
// >>>>>>> END NEW CODE (521ff4297e)
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::long4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type = ::long4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long, 1>
{
  using type = ::ulong1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long, 2>
{
  using type = ::ulong2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long, 3>
{
  using type = ::ulong3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long, 4>
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::ulong4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type = ::ulong4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long long, 1>
{
  using type = ::longlong1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long long, 2>
{
  using type = ::longlong2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long long, 3>
{
  using type = ::longlong3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long long, 4>
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::longlong4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type = ::longlong4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long long, 1>
{
  using type = ::ulonglong1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long long, 2>
{
  using type = ::ulonglong2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long long, 3>
{
  using type = ::ulonglong3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long long, 4>
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::ulonglong4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type = ::ulonglong4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<float, 1>
{
  using type = ::float1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<float, 2>
{
  using type = ::float2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<float, 3>
{
  using type = ::float3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<float, 4>
{
  using type = ::float4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<double, 1>
{
  using type = ::double1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<double, 2>
{
  using type = ::double2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<double, 3>
{
  using type = ::double3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<double, 4>
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::double4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type = ::double4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

#  if _CCCL_HAS_NVFP16()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__half, 2>
{
  using type = ::__half2;
};
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_bfloat16, 2>
{
  using type = ::__nv_bfloat162;
};
#  endif // _CCCL_HAS_NVBF16()

#  if _CCCL_HAS_NVFP8_E4M3()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e4m3, 2>
{
  using type = ::__nv_fp8x2_e4m3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e4m3, 4>
{
  using type = ::__nv_fp8x4_e4m3;
};
#  endif // _CCCL_HAS_NVFP8_E4M3()

#  if _CCCL_HAS_NVFP8_E5M2()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e5m2, 2>
{
  using type = ::__nv_fp8x2_e5m2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e5m2, 4>
{
  using type = ::__nv_fp8x4_e5m2;
};
#  endif // _CCCL_HAS_NVFP8_E5M2()

#  if _CCCL_HAS_NVFP8_E8M0()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e8m0, 2>
{
  using type = ::__nv_fp8x2_e8m0;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e8m0, 4>
{
  using type = ::__nv_fp8x4_e8m0;
};
#  endif // _CCCL_HAS_NVFP8_E8M0()

#  if _CCCL_HAS_NVFP6_E2M3()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp6_e2m3, 2>
{
  using type = ::__nv_fp6x2_e2m3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp6_e2m3, 4>
{
  using type = ::__nv_fp6x4_e2m3;
};
#  endif // _CCCL_HAS_NVFP6_E2M3()

#  if _CCCL_HAS_NVFP6_E3M2()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp6_e3m2, 2>
{
  using type = ::__nv_fp6x2_e3m2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp6_e3m2, 4>
{
  using type = ::__nv_fp6x4_e3m2;
};
#  endif // _CCCL_HAS_NVFP6_E3M2()

#  if _CCCL_HAS_NVFP4_E2M1()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp4_e2m1, 2>
{
  using type = ::__nv_fp4x2_e2m1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp4_e2m1, 4>
{
  using type = ::__nv_fp4x4_e2m1;
};
#  endif // _CCCL_HAS_NVFP4_E2M1()

template <class _Tp, ::cuda::std::size_t _Size>
using vector_type_t = typename vector_type<_Tp, _Size>::type;

// has_vector_type

template <class _Tp, ::cuda::std::size_t _Size>
inline constexpr bool has_vector_type_v = !::cuda::std::is_same_v<vector_type_t<_Tp, _Size>, void>;

template <class _Tp, ::cuda::std::size_t _Size>
using has_vector_type = ::cuda::std::bool_constant<has_vector_type_v<_Tp, _Size>>;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_HAS_CTK()
#endif // _CUDA__TYPE_TRAITS_VECTOR_TYPE_H
