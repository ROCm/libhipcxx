//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _CUDA_STD___EXCEPTION_CUDA_ERROR_H
#define _CUDA_STD___EXCEPTION_CUDA_ERROR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// <<<<<<< OLD CODE from a342ae4ce6 (ac82f7b5b2) - COMMENTED OUT
// #if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
//
// #  include <cuda/std/__exception/exception_macros.h>
// #  include <cuda/std/__exception/terminate.h>
// #  include <cuda/std/source_location>
// =======
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__exception/msg_storage.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/source_location>
// >>>>>>> END NEW CODE (ac82f7b5b2)

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdio>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// <<<<<<< OLD CODE from a342ae4ce6 (ac82f7b5b2) - COMMENTED OUT
// #  if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
// =======
#if _CCCL_HAS_CTK()
// >>>>>>> END NEW CODE (ac82f7b5b2)
using __cuda_error_t = ::cudaError_t;
#else
using __cuda_error_t = int;
#endif

#if !_CCCL_COMPILER(NVRTC)
namespace __detail
{
static char* __format_cuda_error(
  ::cuda::__msg_storage& __msg_buffer,
  const int __status,
  const char* __msg,
  const char* __api                  = nullptr,
  ::cuda::std::source_location __loc = ::cuda::std::source_location::current()) noexcept
{
  ::snprintf(
    __msg_buffer.__buffer,
    512,
    "%s:%d %s%s%s(%d): %s",
    __loc.file_name(),
    __loc.line(),
    __api ? __api : "",
    __api ? " " : "",
#  if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()
    ::cudaGetErrorString(::cudaError_t(__status)),
#  else // ^^^ has CTK or HIP ^^^ / vvv neither CTK nor HIP vvv
    "cudaError",
#  endif // ^^^ neither CTK nor HIP ^^^
    __status,
    __msg);
  return __msg_buffer.__buffer;
}
} // namespace __detail

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
class cuda_error : public ::std::runtime_error
{
public:
  cuda_error(const __cuda_error_t __status,
             const char* __msg,
             const char* __api                  = nullptr,
             ::cuda::std::source_location __loc = ::cuda::std::source_location::current(),
             __msg_storage __msg_buffer         = {}) noexcept
      : ::std::runtime_error(::cuda::__detail::__format_cuda_error(__msg_buffer, __status, __msg, __api, __loc))
      , __status_(__status)
  {}

  [[nodiscard]] auto status() const noexcept -> __cuda_error_t
  {
    return __status_;
  }

private:
  __cuda_error_t __status_;
};
#endif // !_CCCL_COMPILER(NVRTC)

[[noreturn]] _CCCL_API inline void __throw_cuda_error(
  [[maybe_unused]] const __cuda_error_t __status,
  [[maybe_unused]] const char* __msg,
  [[maybe_unused]] const char* __api                  = nullptr,
  [[maybe_unused]] ::cuda::std::source_location __loc = ::cuda::std::source_location::current())
{
  _CCCL_THROW(cuda::cuda_error, __status, __msg, __api, __loc);
}

_CCCL_END_NAMESPACE_CUDA

// <<<<<<< OLD CODE from a342ae4ce6 (ac82f7b5b2) - COMMENTED OUT
// #  include <cuda/std/__cccl/epilogue.h>
//
// #endif // !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
// =======
#include <cuda/std/__cccl/epilogue.h>
// >>>>>>> END NEW CODE (ac82f7b5b2)

#endif // _CUDA_STD___EXCEPTION_CUDA_ERROR_H
