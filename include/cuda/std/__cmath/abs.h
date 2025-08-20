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

#ifndef _LIBCUDACXX___CMATH_ABS_H
#define _LIBCUDACXX___CMATH_ABS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
// <<<<<<< OLD CODE from 8e1b782d9c (e2c8bf22d1) - COMMENTED OUT
// #include <cuda/std/cstdint>
//
// // MSVC and clang cuda need the host side functions included
// #if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG) || defined(_CCCL_HIP_COMPILER)
// #  include <math.h>
// #endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG) || defined(_CCCL_HIP_COMPILER)
// =======
#include <cuda/std/limits>
// >>>>>>> END NEW CODE (e2c8bf22d1)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(builtin_fabs) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FABSF(...) __builtin_fabsf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FABS(...)  __builtin_fabs(__VA_ARGS__)
#  define _CCCL_BUILTIN_FABSL(...) __builtin_fabsl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fabs)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// fabs

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr auto fabs(_Tp __x) noexcept
{
  if constexpr (!numeric_limits<_Tp>::is_signed)
  {
    if constexpr (is_integral_v<_Tp>)
    {
      return static_cast<double>(__x);
    }
    else
    {
      return __x;
    }
  }
  else if constexpr (is_integral_v<_Tp>)
  {
    return __x < 0 ? -static_cast<double>(__x) : static_cast<double>(__x);
  }
  else
  {
#ifdef _CCCL_BUILTIN_FABSF
    if constexpr (is_same_v<_Tp, float>)
    {
      return _CCCL_BUILTIN_FABSF(__x);
    }
    else if constexpr (is_same_v<_Tp, double>)
    {
      return _CCCL_BUILTIN_FABS(__x);
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (is_same_v<_Tp, long double>)
    {
      return _CCCL_BUILTIN_FABSL(__x);
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
#endif // _CCCL_BUILTIN_FABSF
#if _LIBCUDACXX_HAS_NVFP16()
    if constexpr (is_same_v<_Tp, __half>)
    {
      if (!::cuda::std::__cccl_default_is_constant_evaluated())
      {
        return ::__habs(__x);
      }
    }
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
    if constexpr (is_same_v<_Tp, __nv_bfloat16>)
    {
      if (!::cuda::std::__cccl_default_is_constant_evaluated())
      {
        return ::__habs(__x);
      }
    }
#endif // _LIBCUDACXX_HAS_NVBF16()
    const auto __val = ::cuda::std::__fp_get_storage(__x) & __fp_exp_mant_mask_of_v<_Tp>;
    return ::cuda::std::__fp_from_storage<_Tp>(static_cast<__fp_storage_of_t<_Tp>>(__val));
  }
}

[[nodiscard]] _CCCL_API constexpr float fabsf(float __x) noexcept
{
  return ::cuda::std::fabs(__x);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API constexpr long double fabsl(long double __x) noexcept
{
  return ::cuda::std::fabs(__x);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

// abs

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr auto abs(_Tp __x) noexcept
{
  return ::cuda::std::fabs(__x);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_ABS_H
