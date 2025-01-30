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

#ifndef _LIBCUDACXX___CUDA_ATOMIC_PRELUDE_H
#define _LIBCUDACXX___CUDA_ATOMIC_PRELUDE_H

#ifndef __cuda_std__
#error "<__cuda/atomic_prelude> should only be included in from <cuda/std/atomic>"
#endif // __cuda_std__

#if !defined(_LIBCUDACXX_COMPILER_NVRTC) && !defined(_LIBCUDACXX_COMPILER_HIPRTC)
    #include "../cassert" // TRANSITION: Fix transitive includes
    #include <atomic>
    static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "");
    static_assert(ATOMIC_CHAR_LOCK_FREE == 2, "");
    static_assert(ATOMIC_CHAR16_T_LOCK_FREE == 2, "");
    static_assert(ATOMIC_CHAR32_T_LOCK_FREE == 2, "");
    static_assert(ATOMIC_WCHAR_T_LOCK_FREE == 2, "");
    static_assert(ATOMIC_SHORT_LOCK_FREE == 2, "");
    static_assert(ATOMIC_INT_LOCK_FREE == 2, "");
    static_assert(ATOMIC_LONG_LOCK_FREE == 2, "");
    static_assert(ATOMIC_LLONG_LOCK_FREE == 2, "");
    static_assert(ATOMIC_POINTER_LOCK_FREE == 2, "");
    #undef ATOMIC_BOOL_LOCK_FREE
    #undef ATOMIC_BOOL_LOCK_FREE
    #undef ATOMIC_CHAR_LOCK_FREE
    #undef ATOMIC_CHAR16_T_LOCK_FREE
    #undef ATOMIC_CHAR32_T_LOCK_FREE
    #undef ATOMIC_WCHAR_T_LOCK_FREE
    #undef ATOMIC_SHORT_LOCK_FREE
    #undef ATOMIC_INT_LOCK_FREE
    #undef ATOMIC_LONG_LOCK_FREE
    #undef ATOMIC_LLONG_LOCK_FREE
    #undef ATOMIC_POINTER_LOCK_FREE
    #undef ATOMIC_FLAG_INIT
    #undef ATOMIC_VAR_INIT
#endif // _LIBCUDACXX_COMPILER_NVRTC

// pre-define lock free query for heterogeneous compatibility
#ifndef _LIBCUDACXX_ATOMIC_IS_LOCK_FREE
#define _LIBCUDACXX_ATOMIC_IS_LOCK_FREE(__x) (__x <= 8)
#endif

#if !defined(_LIBCUDACXX_COMPILER_NVRTC) && !defined(_LIBCUDACXX_COMPILER_HIPRTC)
#include <thread>
#include <errno.h>
#endif // _LIBCUDACXX_COMPILER_NVRTC

#endif // _LIBCUDACXX___CUDA_ATOMIC_PRELUDE_H
