//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
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

#ifndef _CUDA___FWD_EXECUTION_POLICY_H
#define _CUDA___FWD_EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA() || _CCCL_HIP_COMPILATION()

#  include <cuda/std/__fwd/execution_policy.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

enum __cuda_backend_options : uint16_t
{
  __with_stream          = 1 << 0, ///> Determines whether the policy holds a stream
  __with_memory_resource = 1 << 1, ///> Determines whether the policy holds a memory resource
};

//! @brief Sets the execution backend to cuda
template <uint32_t _Policy>
[[nodiscard]] _CCCL_API constexpr uint32_t __with_cuda_backend() noexcept
{
  constexpr uint32_t __backend_mask{0xFFFF00FF};
  constexpr uint32_t __new_policy =
    (_Policy & __backend_mask) | (static_cast<uint32_t>(__execution_backend::__cuda) << 8);
  return __new_policy;
}

//! @brief Backend specific options of the CUDA backend
template <uint32_t _Policy>
inline constexpr __cuda_backend_options __policy_to_cuda_backend_options =
  static_cast<__cuda_backend_options>((_Policy & uint32_t{0xFFFF0000}) >> 16);

//! @brief Sets a backend specific option
template <uint32_t _Policy, __cuda_backend_options __option>
inline constexpr uint32_t __set_cuda_backend_option =
  _Policy | static_cast<uint32_t>(static_cast<uint32_t>(__option) << 16);

//! @brief Detects whether a given policy holds a user provided stream
template <uint32_t _Policy>
inline constexpr bool __cuda_policy_with_stream =
  static_cast<bool>(__policy_to_cuda_backend_options<_Policy> & __cuda_backend_options::__with_stream);

//! @brief Detects whether a given policy holds a user provided memory resource
template <uint32_t _Policy>
inline constexpr bool __cuda_policy_with_memory_resource =
  static_cast<bool>(__policy_to_cuda_backend_options<_Policy> & __cuda_backend_options::__with_memory_resource);

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA() || _CCCL_HIP_COMPILATION()

#endif // _CUDA___FWD_EXECUTION_POLICY_H
