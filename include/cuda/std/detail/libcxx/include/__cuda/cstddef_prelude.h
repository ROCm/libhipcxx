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

#ifndef _LIBCUDACXX___CUDA_CSTDDEF_PRELUDE_H
#define _LIBCUDACXX___CUDA_CSTDDEF_PRELUDE_H

#ifndef __cuda_std__
#error "<__cuda/cstddef_prelude> should only be included in from <cuda/std/cstddef>"
#endif // __cuda_std__

#if !defined(_LIBCUDACXX_COMPILER_NVRTC) && !defined(_LIBCUDACXX_COMPILER_HIPRTC)
#include <cstddef>
#include <stddef.h>
#else
#define offsetof(type, member) (_CUDA_VSTD::size_t)((char*)&(((type *)0)->member) - (char*)0)
#endif // !defined(_LIBCUDACXX_COMPILER_NVRTC) && !defined(_LIBCUDACXX_COMPILER_HIPRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

typedef decltype(nullptr) nullptr_t;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CUDA_CSTDDEF_PRELUDE_H
