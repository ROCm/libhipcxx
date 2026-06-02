//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

#ifndef _CUDA_STD___FORMAT_FORMAT_ERROR_H
#define _CUDA_STD___FORMAT_FORMAT_ERROR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>

// NOTE(HIP/AMD): Also exclude HIPRTC. Including <stdexcept> here pulls in
// libstdc++ <string> -> <bits/char_traits.h> -> <cstdint> -> system <stdint.h>,
// which under HIPRTC conflicts with the int_fast*_t / uint_fast*_t aliases
// previously defined by cuda/std/cstdint (e.g. int_fast16_t aliased to
// int16_t/short in our header vs. long int in glibc's stdint.h).
#if !defined(_CCCL_COMPILER_HIPRTC)
#  if __cpp_lib_format >= 201907L
#    include <format>
#  else // ^^^ __cpp_lib_format >= 201907L ^^^ / vvv __cpp_lib_format < 201907L vvv
#    include <cuda/std/__host_stdlib/stdexcept>
#  endif // ^^^ __cpp_lib_format < 201907L ^^^
#endif // !defined(_CCCL_COMPILER_HIPRTC)

#include <cuda/std/__cccl/prologue.h>

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

_CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION

#  if __cpp_lib_format >= 201907L
using ::std::format_error;
#  else // ^^^ __cpp_lib_format >= 201907L ^^^ / vvv __cpp_lib_format < 201907L vvv
class _CCCL_TYPE_VISIBILITY_DEFAULT format_error : public ::std::runtime_error
{
public:
  _CCCL_HOST_API explicit format_error(const ::std::string& __s)
      : ::std::runtime_error(__s)
  {}
  _CCCL_HOST_API explicit format_error(const char* __s)
      : ::std::runtime_error(__s)
  {}
  _CCCL_HIDE_FROM_ABI format_error(const format_error&)            = default;
  _CCCL_HIDE_FROM_ABI format_error& operator=(const format_error&) = default;
  _CCCL_HIDE_FROM_ABI virtual ~format_error() noexcept override    = default;
};
#  endif // ^^^ __cpp_lib_format < 201907L ^^^

_CCCL_END_NAMESPACE_CUDA_STD_NOVERSION

#endif // !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[noreturn]] _CCCL_API inline void __throw_format_error([[maybe_unused]] const char* __s)
{
#if _CCCL_HAS_EXCEPTIONS()
  // __s is consumed by the host arm of NV_IF_ELSE_TARGET; the device arm
  // (and the !_CCCL_HAS_EXCEPTIONS() fallback below) discards it. Mark
  // [[maybe_unused]] so the device-only / no-exceptions paths don't trip
  // -Wunused-parameter.
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::cuda::std::format_error(__s);), (::cuda::std::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  ::cuda::std::terminate();
#endif // ^^^ !_CCCL_HAS_EXCEPTIONS() ^^^
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_ERROR_H
