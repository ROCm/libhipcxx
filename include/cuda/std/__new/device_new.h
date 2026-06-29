// -*- C++ -*-
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

#ifndef _CUDA_STD___NEW_DEVICE_NEW_H
#define _CUDA_STD___NEW_DEVICE_NEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// clang-cuda only provides device flavors of operator new if we included <new>
// NOTE(HIP/AMD): hipcc and hiprtc are also clang-based, so the same rule
// applies on HIP: without an explicit '#include <new>', clang does not
// declare the device-side placement new overload, and consumers like
// cuda/__utility/__basic_any/semiregular.h fail with "no matching
// 'operator new' function for non-allocating placement new expression;
// include <new>". Trigger the include on HIP as well.
// NOTE(HIP/AMD): For HIPRTC, the ROCm clang headers at
// /opt/rocm/lib/llvm/lib/clang/22/include/cuda_wrappers/new provide
// __device__ operator new/delete implementations using device malloc/free,
// which allows code using dynamic memory (like cuda::std::seed_seq) to
// compile and link in HIPRTC JIT environments.
#if _CCCL_CUDA_COMPILER(CLANG) || _CCCL_HIP_COMPILATION()
#  include <new>
#endif // _CCCL_CUDA_COMPILER(CLANG) || _CCCL_HIP_COMPILATION()

#endif // _CUDA_STD___NEW_DEVICE_NEW_H
