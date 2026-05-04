//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/std/__cccl/os.h>

// <<<<<<< OLD CODE from cf7bcfc6be (4790c6d706) - COMMENTED OUT
// #if !defined(__CUDACC_RTC__) && !defined(__HIPCC_RTC__)
// =======
// Define these macros to a true value. This tests that CCCL_OS(FOO) is resilient against
// macro-expansion in case the user defines FOO, because if CCCL_OS() expands the macro, then
// the below assertions should fire.
#define APPLE   1
#define LINUX   1
#define WINDOWS 1
#define ANDROID 1
#define QNX     1

#if defined(_WIN32) || defined(_WIN64) || defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__)
#  if CCCL_OS(APPLE)
#    error "macOS detected when it SHOULDN'T be"
#  endif // CCCL_OS(APPLE)
#  if CCCL_OS(LINUX)
#    error "Linux detected when it SHOULDN'T be"
#  endif // CCCL_OS(LINUX)
#  if !CCCL_OS(WINDOWS)
#    error "Windows NOT detected when it SHOULD be"
#  endif // !CCCL_OS(WINDOWS)
#  if CCCL_OS(ANDROID)
#    error "Android detected when it SHOULDN'T be"
#  endif // CCCL_OS(ANDROID)
#  if CCCL_OS(QNX)
#    error "QNX detected when it SHOULDN'T be"
#  endif // CCCL_OS(QNX)
#endif // _WIN32 || _WIN64 || __WIN32__ || __TOS_WIN__ || __WINDOWS__

#if defined(__linux__) || defined(linux) || defined(__linux)
#  if CCCL_OS(APPLE)
#    error "macOS detected when it SHOULDN'T be"
#  endif // CCCL_OS(APPLE)
#  if !CCCL_OS(LINUX)
#    error "Linux NOT detected when it SHOULD be"
#  endif // !CCCL_OS(LINUX)
#  if CCCL_OS(WINDOWS)
#    error "Windows detected when it SHOULDN'T be"
#  endif // CCCL_OS(WINDOWS)
#  ifdef __ANDROID__
#    if !CCCL_OS(ANDROID)
#      error "Android NOT detected when it SHOULD be"
#    endif
#  else // ^^ __ANDROID__ ^^ / vv !__ANDROID__ VV
#    if CCCL_OS(ANDROID)
#      error "Android detected when it SHOULDN'T be"
#    endif // CCCL_OS(ANDROID)
#  endif // __ANDROID__
#  if CCCL_OS(QNX)
#    error "QNX detected when it SHOULDN'T be"
#  endif // CCCL_OS(QNX)
#endif // __linux__ || linux || __linux

#ifdef __ANDROID__
#  if CCCL_OS(APPLE)
#    error "macOS detected when it SHOULDN'T be"
#  endif // CCCL_OS(APPLE)
#  if !CCCL_OS(LINUX)
// Android is also Linux
#    error "Linux NOT detected when it SHOULD be"
#  endif // !CCCL_OS(LINUX)
#  if CCCL_OS(WINDOWS)
#    error "Windows detected when it SHOULDN'T be"
#  endif // CCCL_OS(WINDOWS)
#  if !CCCL_OS(ANDROID)
#    error "Android NOT detected when it SHOULD be"
#  endif // !CCCL_OS(ANDROID)
#  if CCCL_OS(QNX)
#    error "QNX detected when it SHOULDN'T be"
#  endif // CCCL_OS(QNX)
#endif // __ANDROID__

#if defined(__QNX__) || defined(__QNXNTO__)
#  if CCCL_OS(APPLE)
#    error "macOS detected when it SHOULDN'T be"
#  endif // CCCL_OS(APPLE)
#  if CCCL_OS(LINUX)
#    error "Linux detected when it SHOULDN'T be"
#  endif // CCCL_OS(LINUX)
#  if CCCL_OS(WINDOWS)
#    error "Windows detected when it SHOULDN'T be"
#  endif // CCCL_OS(WINDOWS)
#  if CCCL_OS(ANDROID)
#    error "Android detected when it SHOULDN'T be"
#  endif // CCCL_OS(ANDROID)
#  if !CCCL_OS(QNX)
#    error "QNX NOT detected when it SHOULD be"
#  endif // !CCCL_OS(QNX)
#endif // __QNX__ || __QNXNTO__

#if defined(__APPLE__) || defined(__MACH__)
#  if !CCCL_OS(APPLE)
#    error "macOS NOT detected when it SHOULD be"
#  endif // !CCCL_OS(APPLE)
#  if !CCCL_OS(LINUX)
// Unfortunately macOS also triggers linux
#    error "Linux NOT detected when it SHOULD be"
#  endif // !CCCL_OS(LINUX)
#  if CCCL_OS(WINDOWS)
#    error "Windows detected when it SHOULDN'T be"
#  endif // CCCL_OS(WINDOWS)
#  if CCCL_OS(ANDROID)
#    error "Android detected when it SHOULDN'T be"
#  endif // CCCL_OS(ANDROID)
#  if CCCL_OS(QNX)
#    error "QNX detected when it SHOULDN'T be"
#  endif // CCCL_OS(QNX)
#endif // __APPLE__

#if !defined(__CUDACC_RTC__)
// >>>>>>> END NEW CODE (4790c6d706)
#  if _CCCL_OS(WINDOWS)
#    include <windows.h>
#  endif

#  if _CCCL_OS(LINUX)
#    include <unistd.h>
#  endif

#  if _CCCL_OS(ANDROID)
#    include <android/api-level.h>
#  endif

#  if _CCCL_OS(QNX)
#    include <qnx.h>
#  endif
#endif

int main(int, char**)
{
  static_assert(_CCCL_OS(WINDOWS) + _CCCL_OS(LINUX) == 1);
#if _CCCL_OS(ANDROID) || _CCCL_OS(QNX)
  static_assert(_CCCL_OS(LINUX) == 1);
  static_assert(_CCCL_OS(ANDROID) + _CCCL_OS(QNX) == 1);
#endif
#if _CCCL_OS(LINUX)
  static_assert(_CCCL_OS(WINDOWS) == 0);
#endif
#if _CCCL_OS(WINDOWS)
  static_assert(_CCCL_OS(ANDROID) + _CCCL_OS(QNX) + _CCCL_OS(LINUX) == 0);
#endif
  return 0;
}
