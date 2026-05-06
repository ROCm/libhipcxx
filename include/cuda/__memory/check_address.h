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

#ifndef _CUDA___MEMORY_IS_VALID_ADDRESS
#define _CUDA___MEMORY_IS_VALID_ADDRESS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// including cuda/std/limits generates a circular dependency because:
//    numeric_limits -> bit_cast -> cstring -> check_address
// <cuda/std/__utility/cmp.h> also includes cuda/std/limits
#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
// NOTE(HIP/AMD): cuda::device::__is_smem_valid_address_range and the
// cuda::device::__internal_is_address_from helpers it uses are also
// available under HIP via the libhipcxx::__isShared / __isGlobal /
// __isLocal / __isConstant family in <amd/amd_utils.h> (exposed at
// global namespace as ::__isShared etc. on HIP-device pass).
#if _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()
#  include <cuda/__memory/address_space.h>
#  include <cuda/__ptx/instructions/get_sreg.h>
#endif // _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

[[nodiscard]] _CCCL_DEVICE_API inline bool
__is_smem_valid_address_range(const void* __ptr, ::cuda::std::size_t __n) noexcept
{
  if (!::cuda::device::__is_smem_valid_ptr(__ptr))
  {
    return false;
  }
  if (!::cuda::device::__internal_is_address_from(__ptr, ::cuda::device::address_space::shared))
  {
    return false;
  }
  // if __ptr is a shared memory pointer, __ptr + __n must also be a valid shared memory pointer
  if (!::cuda::device::__internal_is_address_from(
        reinterpret_cast<const char*>(__ptr) + __n, ::cuda::device::address_space::shared))
  {
    return false;
  }
  return (__n <= ::cuda::ptx::get_sreg_total_smem_size());
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA

[[nodiscard]] _CCCL_API inline bool __is_valid_address_range(const void* __ptr, ::cuda::std::size_t __n) noexcept
{
  if (__n == 0)
  {
    return false;
  }
  auto __limit = ::cuda::std::uintptr_t{UINTMAX_MAX} - static_cast<::cuda::std::uintptr_t>(__n);
  if (reinterpret_cast<::cuda::std::uintptr_t>(__ptr) > __limit)
  {
    return false;
  }
#if _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()
  // clang-format off
  NV_IF_TARGET(NV_IS_DEVICE, (
    if (::cuda::device::__internal_is_address_from(__ptr, ::cuda::device::address_space::shared) &&
        !::cuda::device::__is_smem_valid_address_range(__ptr, __n))
    {
      return false;
    }
  ));
  // clang-format on
#endif // _CCCL_CUDA_COMPILATION() || _CCCL_HIP_COMPILATION()
  return (__ptr != nullptr);
}

[[nodiscard]] _CCCL_API inline bool __is_valid_address(const void* __ptr) noexcept
{
  return ::cuda::__is_valid_address_range(__ptr, 0);
}

[[nodiscard]] _CCCL_API inline bool
__are_ptrs_overlapping(const void* __ptr_lhs, const void* __ptr_rhs, ::cuda::std::size_t __n) noexcept
{
  auto __ptr1_start = static_cast<const char*>(__ptr_lhs);
  auto __ptr2_start = static_cast<const char*>(__ptr_rhs);
  auto __ptr1_end   = __ptr1_start + __n;
  auto __ptr2_end   = __ptr2_start + __n;
  return __ptr1_start < __ptr2_end && __ptr2_start < __ptr1_end;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_IS_VALID_ADDRESS
