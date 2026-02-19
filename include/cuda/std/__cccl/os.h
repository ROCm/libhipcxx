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

#ifndef __CCCL_OS_H
#define __CCCL_OS_H

// The header provides the following macros to determine the host architecture:
//
// _CCCL_OS(WINDOWS)
// _CCCL_OS(LINUX)
// _CCCL_OS(ANDROID)
// _CCCL_OS(QNX)

// Determine the host compiler and its version
#if defined(_WIN32) || defined(_WIN64) /* _WIN64 for NVRTC */
#  define _CCCL_OS_WINDOWS_() 1
#else
#  define _CCCL_OS_WINDOWS_() 0
#endif

#if defined(__linux__) || ((defined(__CUDACC_RTC__) || defined(__HIPCC_RTC__)) && defined(__LP64__))
#  define _CCCL_OS_LINUX_() 1
#else
#  define _CCCL_OS_LINUX_() 0
#endif

#if defined(__ANDROID__)
#  define _CCCL_OS_ANDROID_() 1
#else
#  define _CCCL_OS_ANDROID_() 0
#endif

#if defined(__QNX__) || defined(__QNXNTO__)
#  define _CCCL_OS_QNX_() 1
#else
#  define _CCCL_OS_QNX_() 0
#endif

#define _CCCL_OS(...) _CCCL_OS_##__VA_ARGS__##_()

#endif // __CCCL_OS_H
