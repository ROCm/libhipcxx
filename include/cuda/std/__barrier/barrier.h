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

#ifndef __LIBCUDACXX___BARRIER_BARRIER_H
#define __LIBCUDACXX___BARRIER_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__barrier/barrier_atomic_wrappers.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__barrier/poll_tester.h>
#include <cuda/std/__new_>
#include <cuda/std/atomic>
#include <cuda/std/chrono>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4324) // structure was padded due to alignment specifier

template <class _CompletionF, thread_scope _Sco = thread_scope_system>
class __barrier_base
{
  __atomic_impl<ptrdiff_t, _Sco> __expected, __arrived;
  _CompletionF __completion;
  __atomic_impl<bool, _Sco> __phase;

public:
  using arrival_token = bool;

private:
  template <typename _Barrier>
  friend class __barrier_poll_tester_phase;
  template <typename _Barrier>
  friend class __barrier_poll_tester_parity;
  template <typename _Barrier>
  _LIBCUDACXX_HIDE_FROM_ABI friend bool __call_try_wait(const _Barrier& __b, typename _Barrier::arrival_token&& __phase);
  template <typename _Barrier>
  _LIBCUDACXX_HIDE_FROM_ABI friend bool __call_try_wait_parity(const _Barrier& __b, bool __parity);

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait(arrival_token __old) const
  {
    return __barrier_atomics::load_acquire<bool, _Sco>(__phase) != __old;
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_parity(bool __parity) const
  {
    return __try_wait(__parity);
  }

public:
  _CCCL_HIDE_FROM_ABI __barrier_base() = default;

  _LIBCUDACXX_HIDE_FROM_ABI __barrier_base(ptrdiff_t __expected, _CompletionF __completion = _CompletionF())
      : __expected(__expected)
      , __arrived(__expected)
      , __completion(__completion)
      , __phase(false)
  {}

  _CCCL_HIDE_FROM_ABI ~__barrier_base() = default;

  __barrier_base(__barrier_base const&)            = delete;
  __barrier_base& operator=(__barrier_base const&) = delete;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI arrival_token arrive(ptrdiff_t __update = 1)
  {
    auto const __old_phase    = __phase.load(memory_order_relaxed);
    auto const __result       = __barrier_atomics::fetch_sub_acq_rel<ptrdiff_t, _Sco>(__arrived, __update) - __update;
    auto const __new_expected = __expected.load(memory_order_relaxed);

    _CCCL_ASSERT(__result >= 0, "");

    if (0 == __result)
    {
      __completion();
      __arrived.store(__new_expected, memory_order_relaxed);
      __barrier_atomics::store_release<bool, _Sco>(__phase, !__old_phase);
      __atomic_notify_all(&__phase.__a, __scope_to_tag<_Sco>{});
    }
    return __old_phase;
  }
  _LIBCUDACXX_HIDE_FROM_ABI void wait(arrival_token&& __old_phase) const
  {
    __phase.wait(__old_phase, memory_order_acquire);
  }
  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_wait()
  {
    wait(arrive());
  }
  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_drop()
  {
    __expected.fetch_sub(1, memory_order_relaxed);
    (void) arrive();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t max() noexcept
  {
    return numeric_limits<ptrdiff_t>::max();
  }
};

_CCCL_DIAG_POP

template <thread_scope _Sco>
class __barrier_base<__empty_completion, _Sco>
{
  static constexpr uint64_t __expected_unit = 1ull;
  static constexpr uint64_t __arrived_unit  = 1ull << 32;
  static constexpr uint64_t __expected_mask = __arrived_unit - 1;
  static constexpr uint64_t __phase_bit     = 1ull << 63;
  static constexpr uint64_t __arrived_mask  = (__phase_bit - 1) & ~__expected_mask;

  __atomic_impl<uint64_t, _Sco> __phase_arrived_expected;

public:
  using arrival_token = uint64_t;

private:
  template <typename _Barrier>
  friend class __barrier_poll_tester_phase;
  template <typename _Barrier>
  friend class __barrier_poll_tester_parity;
  template <typename _Barrier>
  _LIBCUDACXX_HIDE_FROM_ABI friend bool __call_try_wait(const _Barrier& __b, typename _Barrier::arrival_token&& __phase);
  template <typename _Barrier>
  _LIBCUDACXX_HIDE_FROM_ABI friend bool __call_try_wait_parity(const _Barrier& __b, bool __parity);

  static _LIBCUDACXX_HIDE_FROM_ABI constexpr uint64_t __init(ptrdiff_t __count) noexcept
  {
    _CCCL_ASSERT(__count >= 0, "Count must be non-negative.");
    return (((1u << 31) - __count) << 32) | ((1u << 31) - __count);
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_phase(uint64_t __phase) const
  {
    uint64_t const __current = __barrier_atomics::load_acquire<uint64_t, _Sco>(__phase_arrived_expected);
    return ((__current & __phase_bit) != __phase);
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait(arrival_token __old) const
  {
    return __try_wait_phase(__old & __phase_bit);
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_parity(bool __parity) const
  {
    return __try_wait_phase(__parity ? __phase_bit : 0);
  }

public:
  _CCCL_HIDE_FROM_ABI __barrier_base() = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __barrier_base(ptrdiff_t __count, __empty_completion = __empty_completion())
      : __phase_arrived_expected(__init(__count))
  {
    _CCCL_ASSERT(__count >= 0, "");
  }

  _CCCL_HIDE_FROM_ABI ~__barrier_base() = default;

  __barrier_base(__barrier_base const&)            = delete;
  __barrier_base& operator=(__barrier_base const&) = delete;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI arrival_token arrive(ptrdiff_t __update = 1)
  {
    auto const __inc = __arrived_unit * __update;
    auto const __old = __barrier_atomics::fetch_add_acq_rel<uint64_t, _Sco>(__phase_arrived_expected, __inc);
    if ((__old ^ (__old + __inc)) & __phase_bit)
    {
      __phase_arrived_expected.fetch_add((__old & __expected_mask) << 32, memory_order_relaxed);
      __phase_arrived_expected.notify_all();
    }
    return __old & __phase_bit;
  }
  _LIBCUDACXX_HIDE_FROM_ABI void wait(arrival_token&& __phase) const
  {
    _CUDA_VSTD::__cccl_thread_poll_with_backoff(
      __barrier_poll_tester_phase<__barrier_base>(this, _CUDA_VSTD::move(__phase)));
  }
  _LIBCUDACXX_HIDE_FROM_ABI void wait_parity(bool __parity) const
  {
    _CUDA_VSTD::__cccl_thread_poll_with_backoff(__barrier_poll_tester_parity<__barrier_base>(this, __parity));
  }
  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_wait()
  {
    wait(arrive());
  }
  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_drop()
  {
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP_DEVICE_COMPILE__)
    // Use DS atomics for LDS barriers, FLAT atomics for global barriers
    if (__builtin_amdgcn_is_shared(&__phase_arrived_expected)) {
      __attribute__((address_space(3))) uint64_t* __lds_ptr =
        (__attribute__((address_space(3))) uint64_t*)((void*)&__phase_arrived_expected);
      __atomic_fetch_add(__lds_ptr, __expected_unit, __ATOMIC_RELAXED);
    } else {
      __phase_arrived_expected.fetch_add(__expected_unit, memory_order_relaxed);
    }
#else
    __phase_arrived_expected.fetch_add(__expected_unit, memory_order_relaxed);
#endif
    (void) arrive();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t max() noexcept
  {
    return numeric_limits<int32_t>::max();
  }
};

template <class _CompletionF = __empty_completion>
class barrier : public __barrier_base<_CompletionF>
{
public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr barrier(ptrdiff_t __count, _CompletionF __completion = _CompletionF())
      : __barrier_base<_CompletionF>(__count, __completion)
  {}
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __LIBCUDACXX___BARRIER_BARRIER_H
