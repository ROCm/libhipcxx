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

#ifndef _CUDA_STD___HOST_STDLIB_MATH_H
#define _CUDA_STD___HOST_STDLIB_MATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE(HIP/AMD): hipRTC has no host C++ standard library; the only <math.h> on
// the include path is the C header (which defines the math functions as macros).
// Pulling it in here would both fail the C++-compatibility check below and clash
// with cuda::std's own device math. Treat hipRTC like NVRTC and skip the host
// <math.h> include and the macro check entirely.
#if _CCCL_HOSTED()
#  include <math.h>

// Standard C++ library comes with it's own <math.h> C++ compatible header. However, if the include paths are jumbled,
// it might happen that the original C <math.h> is found first. This is a problem because C headers define many of the
// math functions as macros which would change our definitions. So, we check whether any of the functions are defined
// as a macro to distinguish the C++ copatibility header from the C header.
#  if defined(fabs) || defined(fmod) || defined(remainder) || defined(remquo) || defined(fma) || defined(fmax)      \
    || defined(fmin) || defined(fdim) || defined(exp) || defined(exp2) || defined(expm1) || defined(log)            \
    || defined(log10) || defined(log2) || defined(log1p) || defined(pow) || defined(sqrt) || defined(cbrt)          \
    || defined(hypot) || defined(sin) || defined(cos) || defined(tan) || defined(asin) || defined(acos)             \
    || defined(atan) || defined(atan2) || defined(sinh) || defined(cosh) || defined(tanh) || defined(asinh)         \
    || defined(acosh) || defined(atanh) || defined(erf) || defined(erfc) || defined(tgamma) || defined(lgamma)      \
    || defined(ceil) || defined(floor) || defined(trunc) || defined(round) || defined(lround) || defined(llround)   \
    || defined(nearbyint) || defined(rint) || defined(lrint) || defined(llrint) || defined(frexp) || defined(ldexp) \
    || defined(scalbn) || defined(scalbln) || defined(ilogb) || defined(logb) || defined(nextafter)                 \
    || defined(nexttoward) || defined(copysign) || defined(fpclassify) || defined(isfinite) || defined(isinf)       \
    || defined(isnan) || defined(isnormal) || defined(signbit) || defined(isgreater) || defined(isgreaterequal)     \
    || defined(isless) || defined(islessequal) || defined(islessgreater) || defined(isunordered)
#    error \
      "libcu++ requires the C++ compatibility <math.h> header, not the C <math.h> header. Please, check your include paths."
#  endif // math functions defined as macros

#endif // _CCCL_HOSTED()

// NOTE(HIP/AMD): Under hipRTC we do not pull in a host C++ <math.h> (only the C
// <math.h> is reachable). Worse, the HIP runtime headers transitively include
// glibc's C <math.h>, which defines the floating-point classification helpers
// (fpclassify/isnan/isinf/...) and other math functions as function-like
// macros. Those macros corrupt cuda::std's own constexpr math function
// definitions in <cuda/std/__cmath/*> (e.g. `int fpclassify(float)` expands to
// `int __builtin_fpclassify(0,1,4,3,2, float)`). Since this header is included
// by every __cmath/* header before they define their functions, undefine the
// offending C macros here so cuda::std's definitions are used instead. The
// numeric FP_* classification constants from <math.h> are left intact.
#if defined(_CCCL_COMPILER_HIPRTC)
#  undef fpclassify
#  undef isfinite
#  undef isinf
#  undef isnan
#  undef isnormal
#  undef signbit
#  undef isgreater
#  undef isgreaterequal
#  undef isless
#  undef islessequal
#  undef islessgreater
#  undef isunordered
#endif // _CCCL_COMPILER_HIPRTC

#endif // _CUDA_STD___HOST_STDLIB_MATH_H
