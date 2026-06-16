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

#ifndef _CUDA_STD___CMATH_SIGNBIT_H
#define _CUDA_STD___CMATH_SIGNBIT_H

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
#include <cuda/std/limits>

// NOTE(HIP/AMD): under HIPRTC the host <math.h> is unavailable (no compiler
// builtin-include dir in the comgr sandbox) and pulls in libstdc++/glibc headers
// that fail to compile. The impls below use __builtin_* so host <math.h> is not
// needed; guard it out for the RTC path.
#if defined(_CCCL_HIP_COMPILER) && !defined(__HIPCC_RTC__)
#  include <math.h>
#endif // defined(_CCCL_HIP_COMPILER) && !defined(__HIPCC_RTC__)

#include <cuda/std/__cccl/prologue.h>

// NOTE(HIP/AMD): glibc's C <math.h> (pulled in transitively by the HIP runtime
// under hipRTC) defines signbit as a function-like macro that would corrupt the
// cuda::std::signbit definitions below. Undefine it here (after all includes).
#if defined(_CCCL_COMPILER_HIPRTC)
#  undef signbit
#endif // _CCCL_COMPILER_HIPRTC

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool signbit([[maybe_unused]] _Tp __x) noexcept
{
  if constexpr (!numeric_limits<_Tp>::is_signed)
  {
    return false;
  }
  else if constexpr (is_integral_v<_Tp>)
  {
    return __x < 0;
  }
  else
  {
    return ::cuda::std::__fp_get_storage(__x) & __fp_sign_mask_of_v<_Tp>;
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_SIGNBIT_H
