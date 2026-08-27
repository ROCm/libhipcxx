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

#ifndef _CUDA___MEMCPY_ASYNC_HIP_MEMCPY_COMPLETION_H
#define _CUDA___MEMCPY_ASYNC_HIP_MEMCPY_COMPLETION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/async_contract_fulfillment.h>

#include <cuda/__barrier/barrier_hip_block_scope.h>

#include <cuda/__fwd/pipeline.h>
#include <cuda/__memcpy_async/completion_mechanism.h>
#include <cuda/__memcpy_async/is_local_smem_barrier.h>
#include <cuda/__memcpy_async/try_get_barrier_handle.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/cstdint>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// This struct contains functions to defer the completion of a barrier phase
// or pipeline stage until a specific memcpy_async operation *initiated by
// this thread* has completed.

// The user is still responsible for arriving and waiting on (or otherwise
// synchronizing with) the barrier or pipeline barrier to see the results of
// copies from other threads participating in the synchronization object.
struct __memcpy_completion_impl
{
  template <typename _Group>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static async_contract_fulfillment
  __defer(__completion_mechanism __cm,
          _Group const& __group,
          _CUDA_VSTD::size_t __size,
          barrier<::cuda::thread_scope_block>& __barrier)
  {
    // In principle, this is the overload for shared memory barriers. However, a
    // block-scope barrier may also be located in global memory. Therefore, we
    // check if the barrier is a non-smem barrier and handle that separately.
    if (!__is_local_smem_barrier(__barrier))
    {
      return __defer_non_smem_barrier(__cm, __group, __size, __barrier);
    }

    switch (__cm)
    {
      case __completion_mechanism::__async_group:
      {
          (void) __group;
          // Track one completion event for this caller's queued async copy work.
          ::cuda::__add_tx_expectation(__barrier, 1);
          __barrier.__async_arrive_tx();
          return async_contract_fulfillment::async;
      }
      case __completion_mechanism::__async_bulk_group:
      {
        // This completion mechanism should not be used with a shared
        // memory barrier. Or at least, we do not currently envision
        // bulk group to be used with shared memory barriers.
        _CCCL_UNREACHABLE();
      }
      case __completion_mechanism::__mbarrier_complete_tx:
      {
        return async_contract_fulfillment::async;
      }
      case __completion_mechanism::__sync:
      {
        // sync: In this case, we do not need to do anything. The user will have
        // to issue `bar.arrive_wait();` to see the effect of the transaction.
        return async_contract_fulfillment::none;
      }
      default:
      {
        // Get rid of "control reaches end of non-void function":
        _CCCL_UNREACHABLE();
      }
    }
  }

  template <typename _Group, thread_scope _Sco, typename _CompF>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static async_contract_fulfillment __defer(
    __completion_mechanism __cm, _Group const& __group, _CUDA_VSTD::size_t __size, barrier<_Sco, _CompF>& __barrier)
  {
    return __defer_non_smem_barrier(__cm, __group, __size, __barrier);
  }

  template <typename _Group, thread_scope _Sco, typename _CompF>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static async_contract_fulfillment __defer_non_smem_barrier(
    __completion_mechanism __cm, _Group const& __group, _CUDA_VSTD::size_t __size, barrier<_Sco, _CompF>& __barrier)
  {
    // Overload for non-smem barriers.
    switch (__cm)
    {
      case __completion_mechanism::__async_group:
      {
#if defined(__HIP_DEVICE_COMPILE__)
#  if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
        __builtin_amdgcn_s_wait_asynccnt(0);
#  else
      _CCCL_UNREACHABLE();
#  endif
#endif // __HIP_DEVICE_COMPILE__
        return async_contract_fulfillment::async;
      }
      case __completion_mechanism::__mbarrier_complete_tx:
      {
        // Non-smem barriers do not have an mbarrier_complete_tx mechanism..
        _CCCL_UNREACHABLE();
      }
      case __completion_mechanism::__async_bulk_group:
      {
        // This completion mechanism is currently not expected to be used with barriers.
        _CCCL_UNREACHABLE();
      }
      case __completion_mechanism::__sync:
      {
        // sync: In this case, we do not need to do anything.
        return async_contract_fulfillment::none;
      }
      default:
      {
        // Get rid of "control reaches end of non-void function":
        _CCCL_UNREACHABLE();
      }
    }
  }

  template <typename _Group, thread_scope _Sco>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static async_contract_fulfillment
  __defer(__completion_mechanism __cm, _Group const&, _CUDA_VSTD::size_t, pipeline<_Sco>&)
  {
    switch (__cm)
    {
      case __completion_mechanism::__async_group:
        return async_contract_fulfillment::async;
      case __completion_mechanism::__async_bulk_group:
        return async_contract_fulfillment::async;
      case __completion_mechanism::__mbarrier_complete_tx:
        return async_contract_fulfillment::async;
      case __completion_mechanism::__sync:
        return async_contract_fulfillment::none;
      default:
        // Get rid of "control reaches end of non-void function":
        _CCCL_UNREACHABLE();
    }
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___MEMCPY_ASYNC_HIP_MEMCPY_COMPLETION_H
