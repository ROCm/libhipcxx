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

#ifndef _CUDA___NUMERIC_NARROW_H
#define _CUDA___NUMERIC_NARROW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>
// <<<<<<< OLD CODE from 6679bf087e (6f0f385d4f) - COMMENTED OUT
// #include <cuda/std/__floating_point/cuda_fp_types.h>
// =======
#include <cuda/std/__host_stdlib/stdexcept>
// >>>>>>> END NEW CODE (6f0f385d4f)
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! Uses static_cast to cast a value \p __from to type \p _To. \p _To needs to be constructible from \p _From, and \p
//! implement operator!=. This function is intended to show that narrowing and a potential change of the value is
//! intended. Modelled after `gsl::narrow_cast`. See also the C++ Core Guidelines <a
//! href="https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-narrowing">ES.46</a> and <a
//! href="https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-casts-named">ES.49</a>.
template <class _To, class _From>
[[nodiscard]] _CCCL_API constexpr _To
narrow_cast(_From&& __from) noexcept(noexcept(static_cast<_To>(::cuda::std::forward<_From>(__from))))
{
  return static_cast<_To>(::cuda::std::forward<_From>(__from));
}

#if _CCCL_HAS_EXCEPTIONS()
struct narrowing_error : ::std::runtime_error
{
  narrowing_error()
      : ::std::runtime_error("Narrowing error")
  {}
};
#endif // _CCCL_HAS_EXCEPTIONS()

[[noreturn]] _CCCL_API inline void __throw_narrowing_error()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw narrowing_error{};), (::cuda::std::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  ::cuda::std::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
}

#if _CCCL_HIP_COMPILATION()
// NOTE(HIP/AMD): __hip_bfloat16 / __half on HIP do not have direct constructors
// from `(unsigned) long` / `(unsigned) long long`, nor between each other;
// `is_constructible_v<__hip_bfloat16, long>` etc. are false. All half-precision
// types do support construction from / conversion to `double`, so we use
// `double` as a pivot when the direct construction is unavailable. The reverse
// direction (half-precision -> integer) is already covered by the `operator T()`
// member overloads on the half-precision types.
// Tracked: ROCM-23887.
template <class _To, class _From>
inline constexpr bool __narrow_needs_double_pivot_v =
  !::cuda::std::is_constructible_v<_To, _From>
  && (::cuda::std::is_same_v<_To, __nv_bfloat16> || ::cuda::std::is_same_v<_To, __half>);

template <class _To, class _From>
inline constexpr bool __narrow_is_constructible_v =
  ::cuda::std::is_constructible_v<_To, _From> || __narrow_needs_double_pivot_v<_To, _From>;

template <class _To, class _From>
[[nodiscard]] _CCCL_API constexpr _To __narrow_construct(_From __from)
{
  if constexpr (::cuda::std::is_constructible_v<_To, _From>)
  {
    return static_cast<_To>(__from);
  }
  else
  {
    return static_cast<_To>(static_cast<double>(__from));
  }
}
#else // ^^^ _CCCL_HIP_COMPILATION() ^^^ / vvv !_CCCL_HIP_COMPILATION() vvv
template <class _To, class _From>
inline constexpr bool __narrow_is_constructible_v = ::cuda::std::is_constructible_v<_To, _From>;

template <class _To, class _From>
[[nodiscard]] _CCCL_API constexpr _To __narrow_construct(_From __from)
{
  return static_cast<_To>(__from);
}
#endif // !_CCCL_HIP_COMPILATION()

//! Uses static_cast to cast a value \p __from to type \p _To and checks whether the value has changed. \p _To needs
//! to be constructible from \p _From and vice versa, and \p implement operator!=. Throws \ref narrowing_error in host
//! code and traps in device code if the value has changed. Modelled after `gsl::narrow`. See also the C++ Core
//! Guidelines <a href="https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-narrowing">ES.46</a> and <a
//! href="https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-casts-named">ES.49</a>.
template <class _To, class _From>
[[nodiscard]] _CCCL_API constexpr _To narrow(_From __from)
{
  static_assert(__narrow_is_constructible_v<_From, _To>);
  static_assert(__narrow_is_constructible_v<_To, _From>);

  const auto __converted = ::cuda::__narrow_construct<_To>(__from);
  if (::cuda::__narrow_construct<_From>(__converted) != __from)
  {
    ::cuda::__throw_narrowing_error();
  }

  if constexpr (::cuda::std::is_arithmetic_v<_From>)
  {
    if constexpr (::cuda::std::is_signed_v<_From> && !::cuda::std::is_signed_v<_To>)
    {
      if (__from < _From{})
      {
        ::cuda::__throw_narrowing_error();
      }
    }
    if constexpr (!::cuda::std::is_signed_v<_From> && ::cuda::std::is_signed_v<_To>)
    {
      if (__converted < _To{})
      {
        ::cuda::__throw_narrowing_error();
      }
    }
  }
  return __converted;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_NARROW_H
