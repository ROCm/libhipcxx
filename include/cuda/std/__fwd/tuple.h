//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

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

#ifndef _CUDA_STD___FWD_TUPLE_H
#define _CUDA_STD___FWD_TUPLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_HOST_STD_LIB()
_CCCL_BEGIN_NAMESPACE_STD

template <class...>
class tuple;

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB()

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class...>
class _CCCL_TYPE_VISIBILITY_DEFAULT tuple;

#ifndef LIBHIPCXX_NO_LEGACY_COMPAT
// NOTE(HIP/AMD): consumers of libhipcxx <= 3.1 (rocThrust) specialize the class
// template, so the variable template forwards to it. See legacy_compat.h.
template <class>
struct __is_tuple_of_iterator_references
{
  static constexpr bool value = false;
};

template <class _Tp>
inline constexpr bool __is_tuple_of_iterator_references_v = __is_tuple_of_iterator_references<_Tp>::value;
#else // ^^^ !LIBHIPCXX_NO_LEGACY_COMPAT ^^^ / vvv LIBHIPCXX_NO_LEGACY_COMPAT vvv
template <class>
inline constexpr bool __is_tuple_of_iterator_references_v = false;
#endif // LIBHIPCXX_NO_LEGACY_COMPAT

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_TUPLE_H
