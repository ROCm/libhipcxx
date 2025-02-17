// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef _LIBCUDACXX___CUDA_CLIMITS_PRELUDE_H
#define _LIBCUDACXX___CUDA_CLIMITS_PRELUDE_H

#ifndef __cuda_std__
#error "<__cuda/climits_prelude> should only be included in from <cuda/std/climits>"
#endif // __cuda_std__

#if !defined(_LIBCUDACXX_COMPILER_NVRTC) && !defined(_LIBCUDACXX_COMPILER_HIPRTC)
    #include <climits>
    #include <limits.h>
    #include <cstdint>
#else // ^^^ !_LIBCUDACXX_COMPILER_NVRTC ^^^ / vvv _LIBCUDACXX_COMPILER_NVRTC vvv
    #define CHAR_BIT 8

    #define SCHAR_MIN (-128)
    #define SCHAR_MAX 127
    #define UCHAR_MAX 255
    #define __CHAR_UNSIGNED__ ('\xff' > 0) // CURSED
    #if __CHAR_UNSIGNED__
        #define CHAR_MIN 0
        #define CHAR_MAX UCHAR_MAX
    #else
        #define CHAR_MIN SCHAR_MIN
        #define CHAR_MAX SCHAR_MAX
    #endif
    #define SHRT_MIN (-SHRT_MAX - 1)
    #define SHRT_MAX 0x7fff
    #define USHRT_MAX 0xffff
    #define INT_MIN (-INT_MAX - 1)
    #define INT_MAX 0x7fffffff
    #define UINT_MAX 0xffffffff
    #define LONG_MIN (-LONG_MAX - 1)
    #ifdef __LP64__
        #define LONG_MAX LLONG_MAX
        #define ULONG_MAX ULLONG_MAX
    #else
        #define LONG_MAX INT_MAX
        #define ULONG_MAX UINT_MAX
    #endif
    #define LLONG_MIN (-LLONG_MAX - 1)
    #define LLONG_MAX 0x7fffffffffffffffLL
    #define ULLONG_MAX 0xffffffffffffffffUL

    #define __FLT_RADIX__ 2
    #define __FLT_MANT_DIG__ 24
    #define __FLT_DIG__ 6
    #define __FLT_MIN__ 1.17549435082228750796873653722224568e-38F
    #define __FLT_MAX__ 3.40282346638528859811704183484516925e+38F
    #define __FLT_EPSILON__ 1.19209289550781250000000000000000000e-7F
    #define __FLT_MIN_EXP__ (-125)
    #define __FLT_MIN_10_EXP__ (-37)
    #define __FLT_MAX_EXP__ 128
    #define __FLT_MAX_10_EXP__ 38
    #define __FLT_DENORM_MIN__ 1.40129846432481707092372958328991613e-45F
    #define __DBL_MANT_DIG__ 53
    #define __DBL_DIG__ 15
    #define __DBL_MIN__ 2.22507385850720138309023271733240406e-308
    #define __DBL_MAX__ 1.79769313486231570814527423731704357e+308
    #define __DBL_EPSILON__ 2.22044604925031308084726333618164062e-16
    #define __DBL_MIN_EXP__ (-1021)
    #define __DBL_MIN_10_EXP__ (-307)
    #define __DBL_MAX_EXP__ 1024
    #define __DBL_MAX_10_EXP__ 308
    #define __DBL_DENORM_MIN__ 4.94065645841246544176568792868221372e-324

    template<typename _To, typename _From>
    static _LIBCUDACXX_DEVICE _LIBCUDACXX_FORCE_INLINE
    _To __cowchild_cast(_From __from)
    {
        static_assert(sizeof(_From) == sizeof(_To), "");
        union __cast { _From __from; _To __to; };
        __cast __c;
        __c.__from = __from;
        return __c.__to;
    }

    #define __builtin_huge_valf() __cowchild_cast<float>(0x7f800000)
    #define __builtin_nanf(__dummy) __cowchild_cast<float>(0x7fc00000)
    #define __builtin_nansf(__dummy) __cowchild_cast<float>(0x7fa00000)
    #define __builtin_huge_val() __cowchild_cast<double>(0x7ff0000000000000)
    #define __builtin_nan(__dummy) __cowchild_cast<double>(0x7ff8000000000000)
    #define __builtin_nans(__dummy) __cowchild_cast<double>(0x7ff4000000000000)
#endif // _LIBCUDACXX_COMPILER_NVRTC

#endif // _LIBCUDACXX___CUDA_CLIMITS_PRELUDE_H
