//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___BARRIER_BARRIER_ATOMIC_WRAPPERS_H
#define __LIBCUDACXX___BARRIER_BARRIER_ATOMIC_WRAPPERS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/atomic>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Thin wrappers over the atomic operations used in __barrier_base.
//
// On AMD device code, each acquire/release atomic is decomposed into an
// explicit fence plus a relaxed atomic.  This is semantically equivalent:
// a relaxed fetch/store is still a full atomic operation — "relaxed" only
// removes the memory ordering barrier, not the atomicity.  The fence+relaxed
// decomposition may give the AMD compiler slightly more freedom to optimise.
//
// On CUDA device code and host, the wrappers fall back to the standard
// memory_order_acq_rel / memory_order_release / memory_order_acquire orderings.
namespace __barrier_atomics
{

// Replaces fetch_add(acq_rel).  Used in __barrier_base<__empty_completion>::arrive().
template <typename _Tp, thread_scope _Sco>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _Tp fetch_add_acq_rel(__atomic_impl<_Tp, _Sco>& __atom, _Tp __inc)
{
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP_DEVICE_COMPILE__)
  __atomic_thread_fence_cuda(static_cast<__memory_order_underlying_t>(memory_order_release), __scope_to_tag<_Sco>{});

  // Use DS atomics for LDS barriers, FLAT atomics for global barriers
  _Tp val;
  if (__builtin_amdgcn_is_shared(__atom.__a.get()))
  {
    auto const pRaw = reinterpret_cast<void* const>(__atom.__a.get());
    __attribute__((address_space(3))) _Tp* __lds_ptr = (__attribute__((address_space(3))) _Tp*) pRaw;
    val                                              = __atomic_fetch_add(__lds_ptr, __inc, __ATOMIC_RELAXED);
  }
  else
  {
    val = __atom.fetch_add(__inc, memory_order_relaxed);
  }
  __atomic_thread_fence_cuda(static_cast<__memory_order_underlying_t>(memory_order_acquire), __scope_to_tag<_Sco>{});
  return val;
#else
  return __atom.fetch_add(__inc, memory_order_acq_rel);
#endif
}

// Replaces fetch_sub(acq_rel).  Used in __barrier_base<_CompletionF>::arrive().
template <typename _Tp, thread_scope _Sco>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _Tp
fetch_sub_acq_rel(__atomic_impl<_Tp, _Sco>& __atom, _Tp __dec)
{
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP_DEVICE_COMPILE__)
  __atomic_thread_fence_cuda(
    static_cast<__memory_order_underlying_t>(memory_order_release), __scope_to_tag<_Sco>{});
  auto val = __atom.fetch_sub(__dec, memory_order_relaxed);
  __atomic_thread_fence_cuda(
    static_cast<__memory_order_underlying_t>(memory_order_acquire), __scope_to_tag<_Sco>{});
  return val;
#else
  return __atom.fetch_sub(__dec, memory_order_acq_rel);
#endif
}

// Replaces store(release).  Used for the phase flip in __barrier_base<_CompletionF>::arrive().
// A separate fence is required here (not shared with fetch_sub_release above) because
// __completion() may issue stores between the two that must be visible before the phase bit
// is published to waiting threads.
template <typename _Tp, thread_scope _Sco>
_LIBCUDACXX_HIDE_FROM_ABI void store_release(__atomic_impl<_Tp, _Sco>& __atom, _Tp __val)
{
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP_DEVICE_COMPILE__)
  __atomic_thread_fence_cuda(
    static_cast<__memory_order_underlying_t>(memory_order_release), __scope_to_tag<_Sco>{});
  __atom.store(__val, memory_order_relaxed);
#else
  __atom.store(__val, memory_order_release);
#endif
}

// Replaces load(acquire).  Used in __try_wait / __try_wait_phase.
template <typename _Tp, thread_scope _Sco>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _Tp load_acquire(const __atomic_impl<_Tp, _Sco>& __atom)
{
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP_DEVICE_COMPILE__)
  // Use DS_READ for LDS barriers, FLAT_LOAD for global barriers
  _Tp val;
  if (__builtin_amdgcn_is_shared(__atom.__a.get()))
  {
    auto const pRaw = reinterpret_cast<const void*>(__atom.__a.get());
    __attribute__((address_space(3))) const _Tp* __lds_ptr = (__attribute__((address_space(3))) const _Tp*) pRaw;
    val = __atomic_load_n(__lds_ptr, __ATOMIC_RELAXED);
  }
  else
  {
    val = __atom.load(memory_order_relaxed);
  }
  __atomic_thread_fence_cuda(static_cast<__memory_order_underlying_t>(memory_order_acquire), __scope_to_tag<_Sco>{});
  return val;

#else
  return __atom.load(memory_order_acquire);
#endif
}

} // namespace __barrier_atomics

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __LIBCUDACXX___BARRIER_BARRIER_ATOMIC_WRAPPERS_H
