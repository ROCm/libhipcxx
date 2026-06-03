//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef __CCCL_ASSERT_H
#define __CCCL_ASSERT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/attributes.h>
#include <cuda/std/__cccl/builtin.h>
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__cccl/preprocessor.h>

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
#  include <assert.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

#if defined(_DEBUG) || defined(DEBUG)
#  ifndef _CCCL_ENABLE_DEBUG_MODE
#    define _CCCL_ENABLE_DEBUG_MODE
#  endif // !_CCCL_ENABLE_DEBUG_MODE
#endif // _DEBUG || DEBUG

// Automatically enable assertions when debug mode is enabled
#ifdef _CCCL_ENABLE_DEBUG_MODE
#  ifndef CCCL_ENABLE_ASSERTIONS
#    define CCCL_ENABLE_ASSERTIONS
#  endif // !CCCL_ENABLE_ASSERTIONS
#endif // _CCCL_ENABLE_DEBUG_MODE

//! Ensure that we switch on host assertions when all assertions are enabled
#ifndef CCCL_ENABLE_HOST_ASSERTIONS
#  ifdef CCCL_ENABLE_ASSERTIONS
#    define CCCL_ENABLE_HOST_ASSERTIONS
#  endif // CCCL_ENABLE_ASSERTIONS
#endif // !CCCL_ENABLE_HOST_ASSERTIONS

//! Ensure that we switch on device assertions when all assertions are enabled
#ifndef CCCL_ENABLE_DEVICE_ASSERTIONS
#  if defined(CCCL_ENABLE_ASSERTIONS) || defined(__CUDACC_DEBUG__)
#    define CCCL_ENABLE_DEVICE_ASSERTIONS
#  endif // CCCL_ENABLE_ASSERTIONS
#endif // !CCCL_ENABLE_DEVICE_ASSERTIONS

//! Use the different standard library implementations to implement host side asserts
//! _CCCL_ASSERT_IMPL_HOST should never be used directly
#if _CCCL_OS(QNX)
#  define _CCCL_ASSERT_IMPL_HOST(expression, message) ((void) 0)
#elif _CCCL_COMPILER(NVRTC) || defined(_CCCL_COMPILER_HIPRTC) // There is no host standard library in nvrtc/hiprtc
#  define _CCCL_ASSERT_IMPL_HOST(expression, message) ((void) 0)
#elif __has_include(<yvals.h>) && _CCCL_OS(WINDOWS) // Windows uses _STL_VERIFY from <yvals.h>
#  include <yvals.h>
#  define _CCCL_ASSERT_IMPL_HOST(expression, message) _STL_VERIFY(expression, message)
#else // ^^^ MSVC, HIPCC, WIN32 ^^^ / vvv !MSVC STL, HIPCC (Linux/glibc) vvv
#  ifdef NDEBUG
// Reintroduce the __assert_fail / __assert_rtn declaration
extern "C" {
#    if !_CCCL_CUDA_COMPILER(CLANG) && !defined(_CCCL_COMPILER_HIPCC) && !defined(_CCCL_COMPILER_HIPRTC)
_CCCL_HOST_DEVICE
#    endif // !_CCCL_CUDA_COMPILER(CLANG)
#    if _CCCL_OS(APPLE)
void __assert_rtn(const char* __function, const char* __assertion, const char* __file, unsigned int __line) noexcept
  __attribute__((__noreturn__));
#    else // ^^^ _CCCL_OS(APPLE) ^^^ / vvv !_CCCL_OS(APPLE) ^^^
void __assert_fail(const char* __assertion, const char* __file, unsigned int __line, const char* __function) noexcept
  __attribute__((__noreturn__));
#    endif // !_CCCL_OS(APPLE)
}
#  endif // NDEBUG

#  if _CCCL_OS(APPLE)
#    define _CCCL_ASSERT_IMPL_HOST(expression, message)      \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : __assert_rtn(__func__, __FILE__, __LINE__, __message__)
#  elif _CCCL_OS(ANDROID)
#    define _CCCL_ASSERT_IMPL_HOST(expression, message)      \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : __assert2(__FILE__, __LINE__, __func__, message)
#  else // ^^^ _CCCL_OS(APPLE) ^^^ / vvv !_CCCL_OS(APPLE) ^^^
#    define _CCCL_ASSERT_IMPL_HOST(expression, message)      \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : __assert_fail(message, __FILE__, __LINE__, __func__)
#  endif // !_CCCL_OS(APPLE)
#endif // !MSVC STL

//! Use custom implementations with nvcc on device and the host ones with clang-cuda and nvhpc
//! _CCCL_ASSERT_IMPL_DEVICE should never be used directly
#if _CCCL_OS(QNX) || _CCCL_OS(APPLE)
#  define _CCCL_ASSERT_IMPL_DEVICE(expression, message) ((void) 0)
#elif _CCCL_COMPILER(NVRTC) || defined(_CCCL_COMPILER_HIPRTC)
// NOTE(HIP/AMD): Use _wassert on Windows, __assertfail on Linux
#  if defined(_WIN32)
#    define _CCCL_ASSERT_IMPL_DEVICE(expression, message)  \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : _wassert(_CRT_WIDE(message), __FILEW__, __LINE__)
#  elif defined(_CCCL_COMPILER_HIPRTC)
// NOTE(HIP/AMD): HIPRTC's __assertfail (hiprtc_runtime) is a 0-arg __device__-only
// symbol. The 3.3.x backports added host-std-lib iterator helpers (e.g. cuda::next/
// prev) that use _CCCL_ASSERT in __host__ functions. Under device-only HIPRTC the
// assert macro always resolves to the device impl (compilation is device-only), so
// a direct __assertfail() call from a __host__ function fails to type-check, and a
// preprocessor/NV_IF_TARGET guard can't distinguish host vs device there (single
// device TU). Use __host__/__device__ overloading instead: device callers fire the
// real assert (the .runfail tests rely on it), host callers get a no-op.
extern "C" _CCCL_DEVICE void __assertfail();
[[maybe_unused]] _CCCL_DEVICE inline void __cccl_hiprtc_assert_fail() noexcept
{
  __assertfail();
}
[[maybe_unused]] _CCCL_HOST inline void __cccl_hiprtc_assert_fail() noexcept {}
#    define _CCCL_ASSERT_IMPL_DEVICE(expression, message)    \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : ::__cccl_hiprtc_assert_fail()
#  else // NVRTC
#    define _CCCL_ASSERT_IMPL_DEVICE(expression, message)  \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : __assertfail()
#  endif
#elif _CCCL_CUDA_COMPILER(NVCC) || (defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)) //! Use __assert_fail to implement device side asserts
#  if _CCCL_COMPILER(MSVC) || (defined(__HIP_PLATFORM_AMD__) && defined(_WIN32))
#    define _CCCL_ASSERT_IMPL_DEVICE(expression, message)    \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : _wassert(_CRT_WIDE(#message), __FILEW__, __LINE__)
#  elif _CCCL_OS(ANDROID)
#    define _CCCL_ASSERT_IMPL_DEVICE(expression, message)    \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : __assert2(__FILE__, __LINE__, __func__, message)
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#    define _CCCL_ASSERT_IMPL_DEVICE(expression, message)    \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : __assert_fail(message, __FILE__, __LINE__, __func__)
#  endif // !_CCCL_COMPILER(MSVC)
#elif _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()
#  define _CCCL_ASSERT_IMPL_DEVICE(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#else // ^^^ CUDA/HIP compilation ^^^ / vvv !CUDA/HIP compilation vvv
#  define _CCCL_ASSERT_IMPL_DEVICE(expression, message) ((void) 0)
#endif // !CUDA/HIP compilation

//! _CCCL_ASSERT_HOST is enabled conditionally depending on CCCL_ENABLE_HOST_ASSERTIONS
#ifdef CCCL_ENABLE_HOST_ASSERTIONS
#  define _CCCL_ASSERT_HOST(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#else // ^^^ CCCL_ENABLE_HOST_ASSERTIONS ^^^ / vvv !CCCL_ENABLE_HOST_ASSERTIONS vvv
#  define _CCCL_ASSERT_HOST(expression, message) ((void) 0)
#endif // !CCCL_ENABLE_HOST_ASSERTIONS

//! _CCCL_ASSERT_DEVICE is enabled conditionally depending on CCCL_ENABLE_DEVICE_ASSERTIONS
#ifdef CCCL_ENABLE_DEVICE_ASSERTIONS
#  define _CCCL_ASSERT_DEVICE(expression, message) _CCCL_ASSERT_IMPL_DEVICE(expression, message)
#else // ^^^ CCCL_ENABLE_DEVICE_ASSERTIONS ^^^ / vvv !CCCL_ENABLE_DEVICE_ASSERTIONS vvv
#  define _CCCL_ASSERT_DEVICE(expression, message) ((void) 0)
#endif // !CCCL_ENABLE_DEVICE_ASSERTIONS

//! _CCCL_VERIFY is enabled unconditionally and reserved for critical checks that are required to always be on
//! _CCCL_ASSERT is enabled conditionally depending on CCCL_ENABLE_HOST_ASSERTIONS and CCCL_ENABLE_DEVICE_ASSERTIONS
#if _CCCL_CUDA_COMPILER(NVHPC) // NVHPC can't have different behavior for host and device.
                               // The host version of the assert will also work in device code.
#  define _CCCL_VERIFY(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#  if defined(CCCL_ENABLE_HOST_ASSERTIONS) || defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
#    define _CCCL_ASSERT(expression, message) _CCCL_ASSERT_HOST(expression, message)
#  else
#    define _CCCL_ASSERT(expression, message) ((void) 0)
#  endif
#elif _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()
#  if _CCCL_DEVICE_COMPILATION()
#    define _CCCL_VERIFY(expression, message) _CCCL_ASSERT_IMPL_DEVICE(expression, message)
#    define _CCCL_ASSERT(expression, message) _CCCL_ASSERT_DEVICE(expression, message)
#  else // ^^^ _CCCL_DEVICE_COMPILATION() ^^^ / vvv !_CCCL_DEVICE_COMPILATION() vvv
#    define _CCCL_VERIFY(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#    define _CCCL_ASSERT(expression, message) _CCCL_ASSERT_HOST(expression, message)
#  endif // !_CCCL_DEVICE_COMPILATION()
#else // ^^^ CUDA/HIP compilation ^^^ / vvv !CUDA/HIP compilation vvv
#  define _CCCL_VERIFY(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#  define _CCCL_ASSERT(expression, message) _CCCL_ASSERT_HOST(expression, message)
#endif // !CUDA/HIP compilation

#endif // __CCCL_ASSERT_H
