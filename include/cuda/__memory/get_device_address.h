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

#ifndef _CUDA___MEMORY_GET_DEVICE_ADDRESS_H
#define _CUDA___MEMORY_GET_DEVICE_ADDRESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#  include <cuda/__device/device_ref.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/std/__memory/addressof.h>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Returns the device address of the passed \c __device_object
//! @param __device_object the object residing in device memory
//! @warning The user must ensure that the current device is properly set to the device the object was allocated on.
//! @return Valid pointer to the device object
template <class _Tp>
[[nodiscard]] _CCCL_API inline _Tp* get_device_address(_Tp& __device_object)
{
#  if _CCCL_HIP_COMPILATION()
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::cuda::std::addressof(__device_object);), ({
                      void* __device_ptr = nullptr; //
                      _CCCL_TRY_CUDA_API(::hipGetSymbolAddress,
                                         "failed to call cudaGetSymbolAddress in cuda::get_device_address",
                                         &__device_ptr,
                                         __device_object);
                      return static_cast<_Tp*>(__device_ptr);
                    }))
#  else
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::cuda::std::addressof(__device_object);), ({
                      void* __device_ptr = nullptr; //
                      _CCCL_TRY_CUDA_API(::cudaGetSymbolAddress,
                                         "failed to call cudaGetSymbolAddress in cuda::get_device_address",
                                         &__device_ptr,
                                         __device_object);
                      return static_cast<_Tp*>(__device_ptr);
                    }))
#  endif
}

#  if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
//! @brief Returns the address of the passed \c __device_object for the passed \c __device.
//!
//! @param __device_object The object residing in device memory.
//! @param __device The device to query the address for.
//!
//! @return Valid pointer to the device object.
//!
//! @throws cuda::cuda_error if the operation fails.
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API inline _Tp* get_device_address(_Tp& __device_object, device_ref __device)
{
  __ensure_current_context __ctx{__device};
  void* __device_ptr{};
#  if _CCCL_HIP_COMPILATION()
  _CCCL_TRY_CUDA_API(::hipGetSymbolAddress,
                     "failed to call cudaGetSymbolAddress in cuda::get_device_address",
                     &__device_ptr,
                     __device_object);
#  else
  _CCCL_TRY_CUDA_API(::cudaGetSymbolAddress,
                     "failed to call cudaGetSymbolAddress in cuda::get_device_address",
                     &__device_ptr,
                     __device_object);
#  endif
  return static_cast<_Tp*>(__device_ptr);
}
#  endif // !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() || _CCCL_HIP_COMPILATION()

#endif // _CUDA___MEMORY_GET_DEVICE_ADDRESS_H
