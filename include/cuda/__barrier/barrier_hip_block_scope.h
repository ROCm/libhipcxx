// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

/// @file barrier_hip_block_scope.h
/// @brief HIP-native specialisation of `cuda::barrier` for `thread_scope_block`.
///
/// Included by `<cuda/barrier>` when `__HIP_PLATFORM_AMD__` is defined.
/// Provides the full `cuda::barrier<thread_scope_block, __empty_completion>`
/// interface. Shared gfx1250 barriers use an AMD LDS phase object;
/// non-shared and unsupported paths retain the portable software fallback.
/// There are no PTX or NVIDIA-specific code paths in this translation unit.

#ifndef _CUDA___BARRIER_BARRIER_HIP_BLOCK_SCOPE_H
#define _CUDA___BARRIER_BARRIER_HIP_BLOCK_SCOPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier_hip_config.h>
#include <cuda/__fwd/barrier.h>
#include <cuda/__fwd/barrier_native_handle.h>
#include <cuda/__barrier/lds_barrier.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__barrier/barrier_atomic_wrappers.h>
#include <cuda/std/__new_>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/// @brief Dispatch tag carried by every HIP-native barrier specialisation.
///
/// Higher-level dispatch layers test for this tag to distinguish HIP-native
/// specialisations from generic `cuda::barrier` instantiations.
struct __hip_native_tag
{};

// Forward declarations for friend functions
template <class _Barrier>
_CCCL_DEVICE void __add_tx_expectation(_Barrier&, _CUDA_VSTD::uint64_t);

struct __memcpy_completion_impl;

/// @brief HIP-native `cuda::barrier` specialised for `thread_scope_block`.
///
/// Implements the `cuda::barrier` interface for block-scoped synchronisation
/// on AMD GPU targets. Shared gfx1250 barriers use `__lds_barrier_t`
/// as the active phase object; software fallback state is retained for
/// non-shared storage and unsupported targets.
///
/// @note `__shared__` variables of this type must be default-constructed and
///       then explicitly initialised via `init()`, because zero-initialised
///       storage followed by `init()` is the canonical GPU shared-memory
///       pattern and `__shared__` does not support non-trivial constructors.
template <>
class barrier<thread_scope_block, _CUDA_VSTD::__empty_completion>
{
  using __barrier_base = _CUDA_VSTD::__barrier_base<_CUDA_VSTD::__empty_completion, thread_scope_block>;
  __barrier_base __barrier_;

  static constexpr _CUDA_VSTD::uint64_t __phase_bit = 1ull << 63;
  static constexpr _CUDA_VSTD::ptrdiff_t __max_expected_arrivals = 1024;

  friend struct __memcpy_completion_impl;

  _CCCL_DEVICE friend inline _CUDA_VSTD::uint64_t*
  device::_LIBCUDACXX_ABI_NAMESPACE::barrier_native_handle(barrier<thread_scope_block>& __b);

  // LDS phase state for supported shared-memory barriers.
#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
  __lds_barrier_t __tx_barrier;
#endif

#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
  _CCCL_DEVICE bool __is_lds_barrier() const
  {
#  if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_is_shared(reinterpret_cast<const void*>(this));
#  else
    return false;
#  endif
  }

  _CCCL_DEVICE static _CUDA_VSTD::uint64_t __token_from_lds_value(_CUDA_VSTD::uint64_t __value)
  {
    _CUDA_VSTD::uint64_t __phase = __lds_barrier_t::__phase_from_value(__value);
    return (__phase & 1) ? __phase_bit : 0;
  }

  _CCCL_DEVICE void __init_lds_phase(_CUDA_VSTD::ptrdiff_t __expected)
  {
    __tx_barrier.__init_phase(__expected);
  }

  _CCCL_DEVICE _CUDA_VSTD::uint32_t __lds_pending_count() const
  {
    return __tx_barrier.__pending_count();
  }

  _CCCL_DEVICE _CUDA_VSTD::uint64_t __lds_arrive(_CUDA_VSTD::ptrdiff_t __update)
  {
    _CCCL_ASSERT(__update >= 0, "Arrival count update must be non-negative.");
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");

    _CUDA_VSTD::uint64_t __old = __tx_barrier.__arrive_rtn(static_cast<_CUDA_VSTD::uint32_t>(__update));

    return __token_from_lds_value(__old);
  }

  _CCCL_DEVICE bool __try_wait_lds(_CUDA_VSTD::uint64_t __token) const
  {
    _CUDA_VSTD::uint64_t __value = __tx_barrier.__query();

    bool const __complete = __token_from_lds_value(__value) != (__token & __phase_bit);
    if (__complete) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    }
    return __complete;
  }

  _CCCL_DEVICE void __wait_lds(_CUDA_VSTD::uint64_t __token) const
  {
    while (!__try_wait_lds(__token)) {
      __builtin_amdgcn_s_sleep(1);
    }
  }

  _CCCL_DEVICE static _CUDA_VSTD::uint64_t __lds_token_from_parity(bool __parity)
  {
    return __parity ? __phase_bit : 0;
  }

  _CCCL_DEVICE void __lds_arrive_and_drop()
  {
    _CCCL_ASSERT(__builtin_amdgcn_is_shared(reinterpret_cast<const void*>(this)),
                 "arrive_and_drop requires __shared__ allocation on LDS-backed barriers");

    __tx_barrier.__decrement_init_count();

    (void) __lds_arrive(1);
  }
#endif

  _CCCL_DEVICE void __add_tx_pending_count(_CUDA_VSTD::uint64_t __tx_count)
  {
#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
    if (__tx_count == 0) {
      return;
    }
    else {
      // Ordinary barrier-backed async copy bookkeeping requires the barrier in
      // __shared__ memory because __tx_barrier must be in LDS (address space 3)
      // for the hardware async-arrive intrinsic to work.
      _CCCL_ASSERT(__builtin_amdgcn_is_shared(this),
              "async copy bookkeeping requires __shared__ allocation on GFX12+");
      _CCCL_ASSERT(__tx_count <= (__lds_barrier_t::pending_count_max - __lds_pending_count()),
              "Async copy bookkeeping exceeds the LDS barrier pending-count capacity.");

      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
      __tx_barrier.__add_pending_count(__tx_count);

    }
#else
    (void) __tx_count;
#endif
  }

  _CCCL_DEVICE void __async_arrive_tx()
  {
#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
    // __tx_barrier must be in LDS (address space 3); undefined behaviour otherwise.
    _CCCL_ASSERT(__builtin_amdgcn_is_shared(reinterpret_cast<const void*>(this)),
          "async copy completion requires __shared__ allocation on gfx1250");

    __tx_barrier.__async_arrive();
#else
    _CCCL_UNREACHABLE();
#endif
  }

  class __barrier_poll_tester_phase_tx
  {
    barrier const* __barrier_ptr;
    _CUDA_VSTD::uint64_t __phase;

  public:
    _LIBCUDACXX_HIDE_FROM_ABI
    __barrier_poll_tester_phase_tx(barrier const* __barrier_ptr_, _CUDA_VSTD::uint64_t&& __phase_)
        : __barrier_ptr(__barrier_ptr_)
        , __phase(_CUDA_VSTD::move(__phase_))
    {}

    _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool operator()() const
    {
      return __barrier_ptr->__try_wait_phase_tx(__phase);
    }
  };

  class __barrier_poll_tester_parity_tx
  {
    barrier const* __barrier_ptr;
    bool __parity;

  public:
    _LIBCUDACXX_HIDE_FROM_ABI __barrier_poll_tester_parity_tx(barrier const* __barrier_ptr_, bool __parity_)
        : __barrier_ptr(__barrier_ptr_)
        , __parity(__parity_)
    {}

    _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool operator()() const
    {
      return __barrier_ptr->__try_wait_parity_tx(__parity);
    }
  };

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __tx_complete() const
  {
    return true;
  }

  _LIBCUDACXX_HIDE_FROM_ABI void __wait_tx_complete() const
  {
    while (!__tx_complete()) {}
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_tx_complete() const
  {
    if (!__tx_complete()) {
      return false;
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    return true;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_phase_tx(_CUDA_VSTD::uint64_t __phase) const
  {
    if (!_CUDA_VSTD::__call_try_wait(__barrier_, _CUDA_VSTD::move(__phase))) {
      return false;
    }
    return __try_wait_tx_complete();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_parity_tx(bool __parity) const
  {
    if (!_CUDA_VSTD::__call_try_wait_parity(__barrier_, __parity)) {
      return false;
    }
    return __try_wait_tx_complete();
  }

public:
  /// @brief Tag identifying this as the HIP-native specialisation.
  using __hip_dispatch_tag = __hip_native_tag;

  /// @brief Phase token returned by `arrive()` and consumed by `wait()`.
  ///
  /// Encodes the phase bit sampled at the moment of arrival.
  /// Phase 0 (initial): `arrival_token == 0`.
  /// Phase 1: `arrival_token == (1ull << 63)`.
  /// The phase alternates on each barrier completion.
  using arrival_token = _CUDA_VSTD::uint64_t;

  /// @brief Default constructor.
  ///
  /// Leaves the barrier in a zero-initialised state.  Must be followed by
  /// a call to `init()` before any other member function is invoked.
  _CCCL_HIDE_FROM_ABI barrier() = default;

  barrier(const barrier&)            = delete;
  barrier& operator=(const barrier&) = delete;

  /// @brief Constructs a barrier with the given expected arrival count.
  ///
  /// @param __expected  Number of arrivals required to complete one phase.
  ///                    Must satisfy `0 <= __expected <= max()`.
  /// @param             Completion functor (unused; present for API conformance).
  _LIBCUDACXX_HIDE_FROM_ABI explicit barrier(_CUDA_VSTD::ptrdiff_t __expected,
                                             _CUDA_VSTD::__empty_completion = {})
  {
    init(this, __expected);
  }

  /// @brief Initialises a default-constructed barrier in place.
  ///
  /// Intended for use with `__shared__` variables, which are
  /// zero-initialised before the first kernel instruction runs.
  ///
  /// @param __b         Pointer to the barrier to initialise.
  /// @param __expected  Number of arrivals required to complete one phase.
  ///                    Must satisfy `0 <= __expected <= max()`.
  _LIBCUDACXX_HIDE_FROM_ABI friend void init(barrier* __b, _CUDA_VSTD::ptrdiff_t __expected)
  {
    _CCCL_ASSERT(__expected >= 0, "Cannot initialize barrier with negative arrival count");
    _CCCL_ASSERT(__expected <= max(), "Cannot initialize barrier with arrival count greater than max()");
    new (&__b->__barrier_) __barrier_base(__expected);
#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
    __b->__init_lds_phase(__expected);
#else
    (void) __b;
#endif
  }

  /// @brief Arrives at the barrier and returns a phase token.
  ///
  /// Decrements the pending arrival count by `__update`.  When the count
  /// reaches zero the barrier completes and threads blocked in `wait()` are
  /// released.
  ///
  /// @param __update  Number of arrivals to contribute.  Defaults to 1.
  /// @return          A token representing the phase at the time of arrival,
  ///                  to be passed to `wait()`.
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI
  arrival_token arrive(_CUDA_VSTD::ptrdiff_t __update = 1)
  {
#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
    if (__is_lds_barrier()) {
      return __lds_arrive(__update);
    }
#endif
    return __barrier_.arrive(__update);
  }

  /// @brief Blocks until the phase represented by `__token` has completed.
  ///
  /// For LDS-backed phases, waits until public arrivals and internally tracked
  /// asynchronous copy events have both retired the phase.
  ///
  /// @param __token  Token returned by a prior call to `arrive()`.
  ///                 Ownership is transferred; the token must not be used
  ///                 after this call.
  _LIBCUDACXX_HIDE_FROM_ABI
  void wait(arrival_token&& __token) const
  {

#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
    if (__is_lds_barrier()) {
      __wait_lds(__token);
      return;
    }
#endif

    __barrier_.wait(_CUDA_VSTD::move(__token));
    __wait_tx_complete();
  }

  /// @brief Blocks until the barrier reaches the specified phase parity.
  ///
  /// Equivalent to `wait(parity ? (1ull << 63) : 0)` but expressed as a
  /// boolean for user convenience. For LDS-backed phases, waits until public
  /// arrivals and internally tracked asynchronous copy events have both
  /// retired the phase.
  ///
  /// @param __parity  The phase parity to wait for.  `false` waits for
  ///                  phase 0 (even), `true` waits for phase 1 (odd).
  _LIBCUDACXX_HIDE_FROM_ABI
  void wait_parity(bool __parity) const
  {
#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
    if (__is_lds_barrier()) {
      __wait_lds(__lds_token_from_parity(__parity));
      return;
    }
#endif
    __barrier_.wait_parity(__parity);
  __wait_tx_complete();
  }

  /// @brief Attempts to wait for a phase completion with a timeout.
  ///
  /// Returns true if the phase completes before the timeout, false otherwise.
  /// Polls the active phase object until completion or timeout.
  ///
  /// @param __token  The arrival token encoding the phase to wait for.
  /// @param __dur    Maximum duration to wait.
  /// @return true if phase completed, false on timeout.
  template <class _Rep, class _Period>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool
  try_wait_for(arrival_token&& __token, const _CUDA_VSTD::chrono::duration<_Rep, _Period>& __dur) const
  {
    auto __nanosec = _CUDA_VSTD::chrono::duration_cast<_CUDA_VSTD::chrono::nanoseconds>(__dur);
#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
    if (__is_lds_barrier()) {
      arrival_token __token_value = __token;
      if (__nanosec.count() < 1) {
        return __try_wait_lds(__token_value);
      }

      _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start =
        _CUDA_VSTD::chrono::high_resolution_clock::now();
      do {
        if (__try_wait_lds(__token_value)) {
          return true;
        }
      } while (__nanosec > (_CUDA_VSTD::chrono::high_resolution_clock::now() - __start));
      return false;
    }
#endif
    if (__nanosec.count() < 1)
    {
      return __try_wait_phase_tx(_CUDA_VSTD::move(__token));
    }
    return _CUDA_VSTD::__cccl_thread_poll_with_backoff(
      __barrier_poll_tester_phase_tx(this, _CUDA_VSTD::move(__token)),
      __nanosec);
  }

  /// @brief Attempts to wait for a phase completion until a time point.
  ///
  /// Delegates to try_wait_for with the remaining duration.
  ///
  /// @param __token  The arrival token encoding the phase to wait for.
  /// @param __time   Time point deadline.
  /// @return true if phase completed, false on timeout.
  template <class _Clock, class _Duration>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool
  try_wait_until(arrival_token&& __token, const _CUDA_VSTD::chrono::time_point<_Clock, _Duration>& __time) const
  {
    return try_wait_for(_CUDA_VSTD::move(__token), (__time - _Clock::now()));
  }

  /// @brief Attempts to wait for a parity phase completion with a timeout.
  ///
  /// Returns true if the specified parity phase completes before the timeout.
  ///
  /// @param __parity  The phase parity to wait for (false=phase 0, true=phase 1).
  /// @param __dur     Maximum duration to wait.
  /// @return true if phase completed, false on timeout.
  template <class _Rep, class _Period>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool
  try_wait_parity_for(bool __parity, const _CUDA_VSTD::chrono::duration<_Rep, _Period>& __dur) const
  {
    auto __nanosec = _CUDA_VSTD::chrono::duration_cast<_CUDA_VSTD::chrono::nanoseconds>(__dur);
#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
    if (__is_lds_barrier()) {
      _CUDA_VSTD::uint64_t const __token = __lds_token_from_parity(__parity);
      if (__nanosec.count() < 1) {
        return __try_wait_lds(__token);
      }

      _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start =
        _CUDA_VSTD::chrono::high_resolution_clock::now();
      do {
        if (__try_wait_lds(__token)) {
          return true;
        }
      } while (__nanosec > (_CUDA_VSTD::chrono::high_resolution_clock::now() - __start));
      return false;
    }
#endif
    if (__nanosec.count() < 1)
    {
      return __try_wait_parity_tx(__parity);
    }
    return _CUDA_VSTD::__cccl_thread_poll_with_backoff(
      __barrier_poll_tester_parity_tx(this, __parity),
      __nanosec);
  }

  /// @brief Attempts to wait for a parity phase completion until a time point.
  ///
  /// Delegates to try_wait_parity_for with the remaining duration.
  ///
  /// @param __parity  The phase parity to wait for (false=phase 0, true=phase 1).
  /// @param __time    Time point deadline.
  /// @return true if phase completed, false on timeout.
  template <class _Clock, class _Duration>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool
  try_wait_parity_until(bool __parity, const _CUDA_VSTD::chrono::time_point<_Clock, _Duration>& __time) const
  {
    return try_wait_parity_for(__parity, (__time - _Clock::now()));
  }

  /// @brief Arrives at the barrier and waits for the current phase to complete.
  ///
  /// Equivalent to `wait(arrive())`.
  _LIBCUDACXX_HIDE_FROM_ABI
  void arrive_and_wait()
  {
    wait(arrive());
  }

  /// @brief Arrives at the barrier and permanently decrements the expected
  ///        arrival count for all subsequent phases.
  ///
  /// Equivalent to decrementing the expected count by one and then calling
  /// `arrive()`.  The calling thread is not required to call `wait()` on the
  /// current phase and must not call `arrive()` on any subsequent phase.
  ///
  /// @note The expected count used to complete the *current* phase is
  ///       unaffected; only future phases see the decremented threshold.
  _LIBCUDACXX_HIDE_FROM_ABI
  void arrive_and_drop()
  {
#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
    if (__is_lds_barrier()) {
      __lds_arrive_and_drop();
      return;
    }
#endif
    __barrier_.arrive_and_drop();
  }

  /// @brief Returns the maximum expected arrival count accepted by the constructor.
  ///
  /// Matches AMD's practical workgroup arrival limit. LDS-backed barriers use
  /// one current-phase pending-count field shared by arrivals and private
  /// ordinary-copy events; init_count remains the public-arrival reload value.
  ///
  /// @note Declared `static constexpr` so it is callable without an instance.
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _CUDA_VSTD::ptrdiff_t max() noexcept
  {
    return __lds_barrier_t::init_count_max;
  }

  /// @brief Friend function to register internal async-copy bookkeeping.
  ///
  /// Used by ordinary barrier-backed async-copy internals to add pending work
  /// to the active phase before its producer completion arrives.
  template <class _Barrier>
  friend _CCCL_DEVICE void __add_tx_expectation(_Barrier& __b, _CUDA_VSTD::uint64_t __tx_count)
  {
    __b.__add_tx_pending_count(__tx_count);
  }

};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_BARRIER_HIP_BLOCK_SCOPE_H
