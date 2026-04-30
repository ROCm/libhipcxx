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

#ifndef __CCCL_HOST_STD_LIB_H
#define __CCCL_HOST_STD_LIB_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/preprocessor.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#define _CCCL_HOST_STD_LIB_LIBSTDCXX() _CCCL_VERSION_INVALID()
#define _CCCL_HOST_STD_LIB_LIBCXX()    _CCCL_VERSION_INVALID()
#define _CCCL_HOST_STD_LIB_STL()       _CCCL_VERSION_INVALID()

// include a minimal header
// NOTE(HIP/AMD): treat HIPRTC like NVRTC -- do not probe/include the host
// <version>/<ciso646>. NVRTC is a hermetic RTC where __has_include(<version>) is
// already false; HIPRTC's driver is not hermetic, so gate it explicitly. This keeps
// _CCCL_HAS_HOST_STD_LIB() false (no __GLIBCXX__/_LIBCPP_VERSION) and prevents host
// libstdc++ (c++config.h -> initializer_list/stdint/...) from leaking into the
// HIPRTC device translation unit.
#if !defined(_CCCL_COMPILER_HIPRTC)
#  if __has_include(<version>)
#    include <version>
#  elif __has_include(<ciso646>)
#    include <ciso646>
#  endif // ^^^ __has_include(<ciso646>) ^^^
#endif // !defined(_CCCL_COMPILER_HIPRTC)

#define _CCCL_HOST_STD_LIB_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 100 + (_MINOR))
#define _CCCL_HOST_STD_LIB(...)                         _CCCL_VERSION_COMPARE(_CCCL_HOST_STD_LIB_, _CCCL_HOST_STD_LIB_##__VA_ARGS__)

#if _CCCL_HOSTED()
#  if defined(_MSVC_STL_VERSION)
#    undef _CCCL_HOST_STD_LIB_STL
#    define _CCCL_HOST_STD_LIB_STL() (_MSVC_STL_VERSION, 0)
#  elif defined(__GLIBCXX__)
#    undef _CCCL_HOST_STD_LIB_LIBSTDCXX
#    define _CCCL_HOST_STD_LIB_LIBSTDCXX() (_GLIBCXX_RELEASE, 0)
#  elif defined(_LIBCPP_VERSION)
#    undef _CCCL_HOST_STD_LIB_LIBCXX
// since llvm-16, the version scheme has been changed from MMppp to MMmmpp
#    if _LIBCPP_VERSION / 10000 < 2
#      define _CCCL_HOST_STD_LIB_LIBCXX() (_LIBCPP_VERSION / 1000, 0)
#    else
#      define _CCCL_HOST_STD_LIB_LIBCXX() (_LIBCPP_VERSION / 10000, (_LIBCPP_VERSION / 100) % 100)
#    endif
#  endif // ^^^ _LIBCPP_VERSION ^^^
#endif // _CCCL_HOSTED()

#define _CCCL_HAS_HOST_STD_LIB() \
  (_CCCL_HOST_STD_LIB(LIBSTDCXX) || _CCCL_HOST_STD_LIB(LIBCXX) || _CCCL_HOST_STD_LIB(STL))

#endif // __CCCL_HOST_STD_LIB_H
