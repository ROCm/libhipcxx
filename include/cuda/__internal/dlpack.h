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

#ifndef _CUDA___INTERNAL_DLPACK_H
#define _CUDA___INTERNAL_DLPACK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_DLPACK()

#  if __has_include(<dlpack/dlpack.h>)
#    include <dlpack/dlpack.h>
#  elif __has_include(<dlpack.h>)
#    include <dlpack.h>
#  endif

#  define _CCCL_DLPACK_AT_LEAST(_MAJOR, _MINOR) \
    (DLPACK_MAJOR_VERSION > (_MAJOR) || (DLPACK_MAJOR_VERSION == (_MAJOR) && DLPACK_MINOR_VERSION >= (_MINOR)))
#  define _CCCL_DLPACK_BELOW(_MAJOR, _MINOR) (!_CCCL_DLPACK_AT_LEAST(_MAJOR, _MINOR))

#  if DLPACK_MAJOR_VERSION != 1
#    error "Unsupported DLPack version, only version 1 is currently supported"
#  endif // DLPACK_MAJOR_VERSION != 1

// NOTE(HIP/AMD): the DLPack device type for GPU memory is kDLCUDA on NVIDIA but
// kDLROCM on AMD. DLPack has no dedicated ROCm-managed device type, so HIP
// managed memory also maps to kDLROCM.
#  if _CCCL_HIP_COMPILATION()
#    define _CCCL_DLPACK_GPU_DEVICE_TYPE         ::kDLROCM
#    define _CCCL_DLPACK_GPU_MANAGED_DEVICE_TYPE ::kDLROCM
#  else // ^^^ _CCCL_HIP_COMPILATION() ^^^ / vvv !_CCCL_HIP_COMPILATION() vvv
#    define _CCCL_DLPACK_GPU_DEVICE_TYPE         ::kDLCUDA
#    define _CCCL_DLPACK_GPU_MANAGED_DEVICE_TYPE ::kDLCUDAManaged
#  endif // !_CCCL_HIP_COMPILATION()

#endif // _CCCL_HAS_DLPACK()

#endif // _CUDA___INTERNAL_DLPACK_H
