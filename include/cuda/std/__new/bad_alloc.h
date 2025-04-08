// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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

#ifndef _LIBCUDACXX___NEW_BAD_ALLOC_H
#define _LIBCUDACXX___NEW_BAD_ALLOC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>

#if _CCCL_HAS_EXCEPTIONS()
#  include <new>
#endif // !_CCCL_HAS_EXCEPTIONS()

_LIBCUDACXX_BEGIN_NAMESPACE_STD

[[noreturn]] _LIBCUDACXX_HIDE_FROM_ABI void __throw_bad_alloc()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::std::bad_alloc();), (_CUDA_VSTD_NOVERSION::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  _CUDA_VSTD_NOVERSION::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
}

[[noreturn]] _LIBCUDACXX_HIDE_FROM_ABI void __throw_bad_array_new_length()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::std::bad_array_new_length();), (_CUDA_VSTD_NOVERSION::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  _CUDA_VSTD_NOVERSION::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NEW_BAD_ALLOC_H
