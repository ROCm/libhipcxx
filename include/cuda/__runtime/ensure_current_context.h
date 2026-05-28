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

#ifndef _CUDA___RUNTIME_ENSURE_CURRENT_CONTEXT_H
#define _CUDA___RUNTIME_ENSURE_CURRENT_CONTEXT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE(HIP/AMD): on HIP the RAII helper saves the calling thread's
// hipGetDevice() result and restores it on dtor; the CUDA driver-API ctx
// stack has no faithful HIP equivalent so the (CUcontext) overload is
// NV-only. See <libhipcxx/__amd/driver_api.h> for the rationale.
#if (_CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()) && !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

#  include <cuda/__device/device_ref.h>
#  include <cuda/__device/physical_device.h>
#  include <cuda/__driver/driver_api.h>

#  include <cuda/std/__cccl/prologue.h>

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document

_CCCL_BEGIN_NAMESPACE_CUDA

class stream_ref;

//! @brief RAII helper which on construction sets the current context to the specified one.
//! It sets the state back on destruction.
//!
struct [[maybe_unused]] __ensure_current_context
{
  //! @brief Construct a new `__ensure_current_context` object and switch to the primary context of the specified
  //!        device.
  //!
  //! @param new_device The device to switch the context to
  //!
  //! @throws cuda_error if the context switch fails
  _CCCL_HOST_API explicit __ensure_current_context(device_ref __new_device)
  {
#  if _CCCL_HIP_COMPILATION()
    _CCCL_TRY_CUDA_API(::hipGetDevice, "Failed to query current device", &__saved_device_);
    _CCCL_TRY_CUDA_API(::hipSetDevice, "Failed to set current device", __new_device.get());
#  else
    auto __ctx = ::cuda::__physical_devices()[__new_device.get()].__primary_context();
    ::cuda::__driver::__ctxPush(__ctx);
#  endif
  }

#  if !_CCCL_HIP_COMPILATION()
  //! @brief Construct a new `__ensure_current_context` object and switch to the specified
  //!        context. NV-only -- HIP has no driver-API ctx stack.
  //!
  //! @param ctx The context to switch to
  //!
  //! @throws cuda_error if the context switch fails
  _CCCL_HOST_API explicit __ensure_current_context(::CUcontext __ctx)
  {
    ::cuda::__driver::__ctxPush(__ctx);
  }
#  endif

  //! @brief Construct a new `__ensure_current_context` object and switch to the context
  //!        under which the specified stream was created.
  //!
  //! @param stream Stream indicating the context to switch to
  //!
  //! @throws cuda_error if the context switch fails
  _CCCL_HOST_API explicit __ensure_current_context(stream_ref __stream);

  __ensure_current_context(__ensure_current_context&&)                 = delete;
  __ensure_current_context(__ensure_current_context const&)            = delete;
  __ensure_current_context& operator=(__ensure_current_context&&)      = delete;
  __ensure_current_context& operator=(__ensure_current_context const&) = delete;

  //! @brief Destroy the `__ensure_current_context` object and switch back to the original
  //!        context.
  //!
  //! @throws cuda_error if the device switch fails. If the destructor is called
  //!         during stack unwinding, the program is automatically terminated.
  _CCCL_HOST_API ~__ensure_current_context() noexcept(false)
  {
#  if _CCCL_HIP_COMPILATION()
    _CCCL_TRY_CUDA_API(::hipSetDevice, "Failed to restore current device", __saved_device_);
#  else
    // TODO would it make sense to assert here that we pushed and popped the same thing?
    ::cuda::__driver::__ctxPop();
#  endif
  }

#  if _CCCL_HIP_COMPILATION()
private:
  int __saved_device_{};
#  endif
};

_CCCL_END_NAMESPACE_CUDA

#  endif // _CCCL_DOXYGEN_INVOKED

#  include <cuda/std/__cccl/epilogue.h>

#endif // (_CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()) && !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

#endif // _CUDA___RUNTIME_ENSURE_CURRENT_CONTEXT_H
