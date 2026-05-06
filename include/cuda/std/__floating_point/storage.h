//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _CUDA_STD___FLOATING_POINT_STORAGE_H
#define _CUDA_STD___FLOATING_POINT_STORAGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__floating_point/cuda_fp_types.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/traits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr auto __fp_storage_type_impl() noexcept
{
  if constexpr (_Fmt == __fp_format::__fp8_nv_e4m3 || _Fmt == __fp_format::__fp8_nv_e5m2
                || _Fmt == __fp_format::__fp8_nv_e8m0 || _Fmt == __fp_format::__fp6_nv_e2m3
                || _Fmt == __fp_format::__fp6_nv_e3m2 || _Fmt == __fp_format::__fp4_nv_e2m1)
  {
    return uint8_t{};
  }
  else if constexpr (_Fmt == __fp_format::__binary16 || _Fmt == __fp_format::__bfloat16)
  {
    return uint16_t{};
  }
  else if constexpr (_Fmt == __fp_format::__binary32)
  {
    return uint32_t{};
  }
  else if constexpr (_Fmt == __fp_format::__binary64)
  {
    return uint64_t{};
  }
#if _CCCL_HAS_INT128()
  else if constexpr (_Fmt == __fp_format::__fp80_x86 || _Fmt == __fp_format::__binary128)
  {
    return __uint128_t{};
  }
#endif // _CCCL_HAS_INT128()
  else
  {
    static_assert(__always_false_v<decltype(_Fmt)>, "Unsupported floating point format");
  }
}

template <__fp_format _Fmt>
using __fp_storage_t = decltype(__fp_storage_type_impl<_Fmt>());

template <class _Tp>
using __fp_storage_of_t = __fp_storage_t<__fp_format_of_v<_Tp>>;

#if _CCCL_HAS_NVFP16()
struct __cccl_nvfp16_manip_helper : __half
{
  using __half::__x;
};

#  if _CCCL_HIP_COMPILATION()
// NOTE(HIP/AMD): The relevant cross-version axis on HIP's __half is which
// union member of __half_raw the converting constructor
// '__half(const __half_raw&)' reads from, NOT the member's name (the
// member is named 'x' on every ROCm release we know of):
//
//   ROCm <= 7.2 :  __half(__half_raw const& r) : data{r.data} {}    // reads .data
//   ROCm >= ??? :  __half(__half_raw const& r) : __x  {r.x   } {}   // reads .x
//
// (See <hip/amd_detail/amd_hip_fp16.h>'s __half struct in each release.)
//
// To stay constexpr-compatible we have to make whichever union member the
// ctor will read be the *active* member of the __half_raw we hand it.
// Reading an inactive union member at compile time is constant-evaluation-
// invalid in C++17, which is exactly what blocks the upstream
// 'helper.__x = __v' assign-via-friend pattern under HIP.
//
// The SFINAE probe below detects which body the local ROCm headers ship
// by checking whether the *constructor invocation*
// '__half{__half_raw{.x = 0}}' is a constant expression -- which is true
// only when the body reads .x (an active member of the .x-initialised
// raw). On the older body the same expression reads inactive .data and
// the constant evaluation fails. This is a real failure type that gets
// diverted into SFINAE because the failing expression appears as the
// initialiser of a non-type template parameter (which must be a constant
// expression; non-constant initialisers are removed from the candidate
// set instead of triggering a hard error).
template <class _HalfRaw = __half_raw, int = (__half{_HalfRaw{.x = 0}}, 0)>
constexpr true_type __cccl_nvfp16_ctor_reads_x_probe(int);
template <class = __half_raw>
constexpr false_type __cccl_nvfp16_ctor_reads_x_probe(...);

template <class _HalfRaw = __half_raw>
inline constexpr bool __cccl_nvfp16_ctor_reads_x_v =
  decltype(::cuda::std::__cccl_nvfp16_ctor_reads_x_probe<_HalfRaw>(0))::value;

// One unified helper: '__cccl_make_nvfp16_raw(v)' returns a properly-
// initialised __half_raw using whichever activation pattern lets
// __half(const __half_raw&) consume it constexpr-compatibly on the
// current ROCm release. Two SFINAE-guarded overloads pick between them
// automatically; callers do not need an if/else.
template <class _HalfRaw = __half_raw, enable_if_t<__cccl_nvfp16_ctor_reads_x_v<_HalfRaw>, int> = 0>
[[nodiscard]] _CCCL_API constexpr _HalfRaw __cccl_make_nvfp16_raw(unsigned short __v) noexcept
{
  // Newer ROCm: __half's ctor reads .x (unsigned short) from the raw.
  // Make .x the active union member so the read is constant-valid.
  return _HalfRaw{.x = __v};
}
template <class _HalfRaw = __half_raw, enable_if_t<!__cccl_nvfp16_ctor_reads_x_v<_HalfRaw>, int> = 0>
[[nodiscard]] _CCCL_API constexpr _HalfRaw __cccl_make_nvfp16_raw(unsigned short __v) noexcept
{
  // ROCm <= 7.2: __half's ctor reads .data (_Float16) from the raw.
  // Bit-cast the storage into the _Float16 union member so .data is the
  // active member (constexpr-friendly since clang v9 because both are
  // scalar types of equal size).
  return _HalfRaw{.data = __builtin_bit_cast(_Float16, __v)};
}

// Helper exposing __half's protected union members so we can extract the
// 16-bit storage from a constexpr __half value. Mirror of
// __cccl_nvfp16_manip_helper but exposing 'data' (current ROCm) too so
// __cccl_get_nvfp16_storage() below can use whichever member is active.
struct __cccl_nvfp16_extract_helper : __half
{
  using __half::__x;
#    if !_CCCL_HIP_COMPILATION()
  // 'data' is HIP-only; on NVIDIA, __half does not expose a 'data' member
  // and the upstream path '.__x' suffices.
#    else
  using __half::data;
#    endif
};

// One unified helper: '__cccl_get_nvfp16_storage(v)' returns the 16-bit
// representation of a __half value in a constexpr-compatible way on the
// current ROCm release. Two SFINAE-guarded overloads pick between the
// __x and data union members automatically -- the active one is the
// member __half's ctor wrote to (which is the one the corresponding
// __cccl_make_nvfp16_raw above made active in the input raw).
template <class _HalfRaw = __half_raw, enable_if_t<__cccl_nvfp16_ctor_reads_x_v<_HalfRaw>, int> = 0>
[[nodiscard]] _CCCL_API constexpr unsigned short __cccl_get_nvfp16_storage(__half __v) noexcept
{
  // Newer ROCm: ctor sets '__x{r.x}' so '__half::__x' is the active
  // member -- read it directly.
  return __cccl_nvfp16_extract_helper{__v}.__x;
}
template <class _HalfRaw = __half_raw, enable_if_t<!__cccl_nvfp16_ctor_reads_x_v<_HalfRaw>, int> = 0>
[[nodiscard]] _CCCL_API constexpr unsigned short __cccl_get_nvfp16_storage(__half __v) noexcept
{
  // ROCm <= 7.2: ctor sets 'data{r.data}' (the _Float16 member). Read
  // 'data' and bit_cast back to the 16-bit storage scalar.
  return __builtin_bit_cast(unsigned short, __cccl_nvfp16_extract_helper{__v}.data);
}
#  endif // _CCCL_HIP_COMPILATION()
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
struct __cccl_nvbf16_manip_helper : __nv_bfloat16
{
  using __nv_bfloat16::__x;
};
#endif // _CCCL_HAS_NVBF16()

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_from_storage(__fp_storage_of_t<_Tp> __v) noexcept
{
  if constexpr (__is_std_fp_v<_Tp> || __is_ext_compiler_fp_v<_Tp>)
  {
    return ::cuda::std::bit_cast<_Tp>(__v);
  }
  else if constexpr (__is_ext_cccl_fp_v<_Tp>)
  {
    _Tp __ret{};
    __ret.__storage_ = __v;
    return __ret;
  }
#if _CCCL_HAS_NVFP16()
  else if constexpr (is_same_v<_Tp, __half>)
  {
    // NOTE(HIP/AMD): HIP's __half embeds a protected union, so the upstream
    // path 'helper.__x = __v' assigns to the *inactive* union member, which
    // is disallowed in a constant expression. Construct a __half_raw with
    // the 16-bit storage member directly initialised, and let
    // __half(const __half_raw&) consume it constexpr-compatibly.
    // The storage member is named '__x' on future ROCm releases or 'x' on
    // ROCm <=7.2; __cccl_make_nvfp16_raw() detects which automatically.
#  if _CCCL_HIP_COMPILATION()
    return __half{__cccl_make_nvfp16_raw(__v)};
#  else
    __cccl_nvfp16_manip_helper __helper{};
    __helper.__x = __v;
    return __helper;
#  endif
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (is_same_v<_Tp, __nv_bfloat16>)
  {
    __cccl_nvbf16_manip_helper __helper{};
    __helper.__x = __v;
    return __helper;
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (is_same_v<_Tp, __nv_fp8_e4m3>)
  {
    __nv_fp8_e4m3 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (is_same_v<_Tp, __nv_fp8_e5m2>)
  {
    __nv_fp8_e5m2 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (is_same_v<_Tp, __nv_fp8_e8m0>)
  {
    __nv_fp8_e8m0 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (is_same_v<_Tp, __nv_fp6_e2m3>)
  {
    _CCCL_ASSERT((__v & 0xc0u) == 0u, "Invalid __nv_fp6_e2m3 storage value");
    __nv_fp6_e2m3 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (is_same_v<_Tp, __nv_fp6_e3m2>)
  {
    _CCCL_ASSERT((__v & 0xc0u) == 0u, "Invalid __nv_fp6_e3m2 storage value");
    __nv_fp6_e3m2 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (is_same_v<_Tp, __nv_fp4_e2m1>)
  {
    _CCCL_ASSERT((__v & 0xf0u) == 0u, "Invalid __nv_fp4_e2m1 storage value");
    __nv_fp4_e2m1 __ret{};
    __ret.__x = __v;
    return __ret;
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating point format");
  }
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES((!is_same_v<_Up, __fp_storage_of_t<_Tp>>) )
_CCCL_API constexpr _Tp __fp_from_storage(const _Up& __v) noexcept = delete;

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __fp_storage_of_t<_Tp> __fp_get_storage(_Tp __v) noexcept
{
  if constexpr (__is_std_fp_v<_Tp> || __is_ext_compiler_fp_v<_Tp>)
  {
    return ::cuda::std::bit_cast<__fp_storage_of_t<_Tp>>(__v);
  }
  else if constexpr (__is_ext_cccl_fp_v<_Tp>)
  {
    return __v.__storage_;
  }
#if _CCCL_HAS_NVFP16()
  else if constexpr (is_same_v<_Tp, __half>)
  {
    // NOTE(HIP/AMD): see __cccl_get_nvfp16_storage above for the constexpr
    // WAR rationale. Mirrors __fp_from_storage<__half>.
#  if _CCCL_HIP_COMPILATION()
    return __cccl_get_nvfp16_storage(__v);
#  else
    return __cccl_nvfp16_manip_helper{__v}.__x;
#  endif
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (is_same_v<_Tp, __nv_bfloat16>)
  {
    return __cccl_nvbf16_manip_helper{__v}.__x;
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (is_same_v<_Tp, __nv_fp8_e4m3>)
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (is_same_v<_Tp, __nv_fp8_e5m2>)
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (is_same_v<_Tp, __nv_fp8_e8m0>)
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (is_same_v<_Tp, __nv_fp6_e2m3>)
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (is_same_v<_Tp, __nv_fp6_e3m2>)
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (is_same_v<_Tp, __nv_fp4_e2m1>)
  {
    return __v.__x;
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating point format");
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FLOATING_POINT_STORAGE_H
