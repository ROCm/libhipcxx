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

#ifndef _CUDA_STD___INTERNAL_PSTL_CONFIG_H
#define _CUDA_STD___INTERNAL_PSTL_CONFIG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

// NOTE(HIP/AMD): _CCCL_HAS_BACKEND_CUDA() is deliberately kept CUDA-only so
// that the NVIDIA-CUB backend include paths are never activated on HIP.  Each
// enabled pstl algorithm frontend instead pulls in the hipCUB-backed __hipcub.h
// shim behind _CCCL_HAS_BACKEND_HIP() (below).
#define _CCCL_HAS_BACKEND_CUDA() _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
#define _CCCL_HAS_BACKEND_OMP()  0
#define _CCCL_HAS_BACKEND_TBB()  0

// NOTE(HIP/AMD): the hipCUB-backed pstl device backend requires the HIP host
// runtime (hipStream_t / hipMemcpyKind / hipSetDevice, ...) plus the host stdlib
// headers that <hipcub/hipcub.hpp> + rocPRIM pull in (<atomic>, <stdint.h>, ...),
// none of which exist under HIPRTC (device-only COMGR compilation -- the stdlib
// headers also clash with libhipcxx's own <cstdint>). So the backend is available
// only for full hipcc compilation, NOT HIPRTC -- exactly mirroring the way
// _CCCL_HAS_BACKEND_CUDA() excludes NVRTC. Use this instead of a bare
// _CCCL_HIP_COMPILATION() wherever a hipCUB include path is gated.
#define _CCCL_HAS_BACKEND_HIP() (_CCCL_HIP_COMPILATION() && !defined(_CCCL_COMPILER_HIPRTC))

// HIP uses the __cuda execution backend enum value (already gated by
// _CCCL_HIP_COMPILATION() in __fwd/execution_policy.h) and routes dispatch
// through the hipCUB-backed specialisations added to each enabled algorithm.
// Widening _CCCL_HAS_PSTL_BACKEND() to include HIP makes the pstl algorithm
// declarations in <cuda/std/execution> visible when building with hipcc, without
// activating any NVIDIA-CUB include path (those remain behind _CCCL_HAS_BACKEND_CUDA()).
#define _CCCL_HAS_PSTL_BACKEND() \
  (_CCCL_HAS_BACKEND_CUDA() || _CCCL_HAS_BACKEND_OMP() || _CCCL_HAS_BACKEND_TBB() || _CCCL_HAS_BACKEND_HIP())

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___INTERNAL_PSTL_CONFIG_H
