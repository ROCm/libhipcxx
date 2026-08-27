// -*- C++ -*-
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

#ifndef _CUDA__MEMCPY_ASYNC_DISPATCH_HIP_MEMCPY_ASYNC_H_
#define _CUDA__MEMCPY_ASYNC_DISPATCH_HIP_MEMCPY_ASYNC_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier_hip_config.h>

#if defined(__HIP_DEVICE_COMPILE__) && _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
#  include <hip/amd_detail/amd_hip_cooperative_groups_memcpy.h> //(HIP/AMD TODO): update once dependent PR lands
#endif
#include <cuda/__memcpy_async/completion_mechanism.h>
#include <cuda/__memcpy_async/cp_async_fallback.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/***********************************************************************
 * cuda::memcpy_async dispatch
 *
 * The dispatch mechanism takes all the arguments and dispatches to the
 * fastest asynchronous copy mechanism available.
 *
 * It returns a __completion_mechanism that indicates which completion mechanism
 * was used by the copy mechanism. This value can be used by the sync object to
 * further synchronize if necessary.
 *
 ***********************************************************************/

_CCCL_NODISCARD _CCCL_DEVICE inline bool __hip_memcpy_async_is_shared(char const* __ptr)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_is_shared((const __attribute__((address_space(0))) void*) __ptr);
#else // __HIP_DEVICE_COMPILE__
  (void) __ptr;
  return false;
#endif // __HIP_DEVICE_COMPILE__
}

_CCCL_NODISCARD _CCCL_DEVICE inline bool
__hip_memcpy_async_has_accelerated_direction(char* __dest_char, char const* __src_char)
{
#if defined(__HIP_DEVICE_COMPILE__) && _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
  bool const __src_is_shared = __hip_memcpy_async_is_shared(__src_char);
  bool const __dst_is_shared = __hip_memcpy_async_is_shared(__dest_char);
  return __src_is_shared != __dst_is_shared;
#else
  (void) __dest_char;
  (void) __src_char;
  return false;
#endif
}

template <_CUDA_VSTD::size_t _Align, typename _Group>
_CCCL_NODISCARD _CCCL_DEVICE inline __completion_mechanism __dispatch_memcpy_async_any_to_any(
  _Group const& __group,
  char* __dest_char,
  char const* __src_char,
  _CUDA_VSTD::size_t __size,
  _CUDA_VSTD::uint32_t __allowed_completions,
  _CUDA_VSTD::uint64_t* __bar_handle)
{
  const bool __can_use_complete_tx = __allowed_completions & uint32_t(__completion_mechanism::__mbarrier_complete_tx);
  const bool __can_use_async_group = __allowed_completions & uint32_t(__completion_mechanism::__async_group);

#if defined(__HIP_DEVICE_COMPILE__) && _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
  bool const __src_is_shared = __hip_memcpy_async_is_shared(__src_char);
  bool const __dst_is_shared = __hip_memcpy_async_is_shared(__dest_char);
  bool const __has_accelerated_direction = __src_is_shared != __dst_is_shared;

  if((__can_use_complete_tx || __can_use_async_group) && __has_accelerated_direction)
  {
    if (__size == 0) {
      return __completion_mechanism::__sync;
    }

    // We have total size in bytes: __size
    // Total count of threads: __group.size()
    // Each thread will have to do __size / __group.size() bytes copy in lockstep
    size_t group_size = __group.size();
    size_t bytes_per_thread = __size / group_size;
    if (__src_is_shared && !__dst_is_shared && bytes_per_thread > 0) {
      cooperative_groups::details::accelerated_memcpy_lds_to_global(__dest_char, __src_char, bytes_per_thread * __group.thread_rank(),
                                                bytes_per_thread);
    } else if (!__src_is_shared && __dst_is_shared && bytes_per_thread > 0) {
      cooperative_groups::details::accelerated_memcpy_global_to_lds(__dest_char, __src_char, bytes_per_thread * __group.thread_rank(),
                                                bytes_per_thread);
    }

    // Now we handle data that could not be copied alongside all threads
    // example: user asked to copy 33 bytes on 32 threads, each thread will do 1 byte async-copy in
    // lock-step but for the last 1 byte we need to manually handle it and enqueue the memcpy
    size_t bytes_copied = bytes_per_thread * group_size;
    if (__group.thread_rank() == 0 && __size > bytes_copied) {
      if (__src_is_shared && !__dst_is_shared) {
        cooperative_groups::details::accelerated_memcpy_lds_to_global(__dest_char, __src_char, bytes_copied, __size - bytes_copied);
      } else if (!__src_is_shared && __dst_is_shared) {
        cooperative_groups::details::accelerated_memcpy_global_to_lds(__dest_char, __src_char, bytes_copied, __size - bytes_copied);
      }
    }
  }
  else
#endif
  {
    __cp_async_fallback_mechanism<_Align>(__group, __dest_char, __src_char, __size);
    return __completion_mechanism::__sync;
  }

#if defined(__HIP_DEVICE_COMPILE__) && _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
  if(__bar_handle != nullptr && (__can_use_complete_tx))
  {
    __attribute__((address_space(3))) _CUDA_VSTD::int64_t* __tmpHandle = (__attribute__((address_space(3))) _CUDA_VSTD::int64_t*)(__bar_handle);

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_ds_atomic_async_barrier_arrive_b64(__tmpHandle);

    return __completion_mechanism::__mbarrier_complete_tx;
  }
#else
  (void) __bar_handle;
#endif

  if(__can_use_async_group)
  {
    return __completion_mechanism::__async_group;
  }

  return __completion_mechanism::__sync;
}

// __dispatch_memcpy_async is the internal entry point for dispatching to the correct memcpy_async implementation.
template <_CUDA_VSTD::size_t _Align, typename _Group>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __completion_mechanism __dispatch_memcpy_async(
  _Group const& __group,
  char* __dest_char,
  char const* __src_char,
  _CUDA_VSTD::size_t __size,
  _CUDA_VSTD::uint32_t __allowed_completions,
  _CUDA_VSTD::uint64_t* __bar_handle)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE_LIBHIPCXX,
    (
        return __dispatch_memcpy_async_any_to_any<_Align>(
          __group, __dest_char, __src_char, __size, __allowed_completions, __bar_handle);
    ),
    (
      // Host code path:
      if (__group.thread_rank() == 0) {
        _CUDA_VSTD::memcpy(__dest_char, __src_char, __size);
      } return __completion_mechanism::__sync;));
}

template <_CUDA_VSTD::size_t _Align, typename _Group>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __completion_mechanism __dispatch_memcpy_async(
  _Group const& __group,
  char* __dest_char,
  char const* __src_char,
  _CUDA_VSTD::size_t __size,
  _CUDA_VSTD::uint32_t __allowed_completions)
{
  _CCCL_ASSERT(!(__allowed_completions & uint32_t(__completion_mechanism::__mbarrier_complete_tx)),
               "Cannot allow mbarrier_complete_tx completion mechanism when not passing a barrier. ");
  return __dispatch_memcpy_async<_Align>(__group, __dest_char, __src_char, __size, __allowed_completions, nullptr);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA__MEMCPY_ASYNC_DISPATCH_HIP_MEMCPY_ASYNC_H_
