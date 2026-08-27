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

/// @file lds_barrier.h
/// @brief AMD LDS barrier helper class for phase tracking.
///
/// Wraps the AMD LDS barrier 64-bit packed word with bitfield accessors
/// for pending_count, phase, and init_count.

#ifndef _CUDA___BARRIER_LDS_BARRIER_H
#define _CUDA___BARRIER_LDS_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier_hip_config.h>
#include <cuda/__atomic/atomic.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
extern "C" _CCCL_DEVICE void llvm_amdgcn_s_wait_dscnt(unsigned short)
  __asm("llvm.amdgcn.s.wait.dscnt");
#endif // _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT

/// @brief AMD LDS barrier helper with bitfield layout
///
/// 64-bit packed word matching AMD LDS barrier hardware layout:
/// - [28:0]  pending_count (29 bits)
/// - [31:29] phase (3 bits, values 0-7)
/// - [47:32] init_count (16 bits)
/// - [63:48] zeros (16 bits, reserved)
struct alignas(8) __lds_barrier_t
{
  static constexpr int WIDTH = 29;
  static constexpr int pending_count_width = WIDTH;
  static constexpr int init_count_width    = 16;
  static constexpr _CUDA_VSTD::uint32_t pending_count_max = (_CUDA_VSTD::uint32_t{1} << pending_count_width) - 1;
  static constexpr _CUDA_VSTD::uint64_t combined_count_max = _CUDA_VSTD::uint64_t{1} << pending_count_width;
  static constexpr _CUDA_VSTD::uint32_t init_count_max = (_CUDA_VSTD::uint32_t{1} << init_count_width) - 1;

  union {
    _CUDA_VSTD::uint64_t value;
    struct {
      _CUDA_VSTD::uint64_t pending_count : WIDTH;         // [28:0] arrive(N)
      _CUDA_VSTD::uint64_t phase         : (32 - WIDTH);  // [31:29] phase modified if last to arrive
      _CUDA_VSTD::uint64_t init_count    : init_count_width; // [47:32] next-phase public arrival count
      _CUDA_VSTD::uint64_t zeros         : 16;            // [63:48] 
    };
  };

  /// @brief Set the init_count bitfield
  /// @param __count Value to write to init_count (16-bit)
  _CCCL_HOST_DEVICE void set_init_count(_CUDA_VSTD::uint16_t __count)
  {
    init_count = __count;
  }

  /// @brief Set the pending_count bitfield
  /// @param __count Value to write to pending_count (29-bit)
  _CCCL_HOST_DEVICE void set_pending_count(_CUDA_VSTD::uint32_t __count)
  {
    pending_count = __count;
  }

#if _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
  _CCCL_FORCEINLINE _CCCL_DEVICE __attribute__((address_space(3))) _CUDA_VSTD::uint64_t* __lds_word_ptr()
  {
    return (__attribute__((address_space(3))) _CUDA_VSTD::uint64_t*)(&value);
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE __attribute__((address_space(3))) _CUDA_VSTD::uint64_t* __lds_word_ptr() const
  {
    return (__attribute__((address_space(3))) _CUDA_VSTD::uint64_t*)(&const_cast<__lds_barrier_t*>(this)->value);
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE static _CUDA_VSTD::uint64_t __phase_from_value(_CUDA_VSTD::uint64_t __value)
  {
    __lds_barrier_t __barrier = {};
    __barrier.value = __value;
    return __barrier.phase;
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE static _CUDA_VSTD::uint32_t __pending_count_from_value(_CUDA_VSTD::uint64_t __value)
  {
    __lds_barrier_t __barrier = {};
    __barrier.value = __value;
    return static_cast<_CUDA_VSTD::uint32_t>(__barrier.pending_count);
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE void __init_phase(_CUDA_VSTD::ptrdiff_t __expected)
  {
    value = 0;

    if (__expected > 0) {
      _CUDA_VSTD::uint64_t __init_val = static_cast<_CUDA_VSTD::uint64_t>(__expected - 1);
      set_pending_count(static_cast<_CUDA_VSTD::uint32_t>(__init_val));
      set_init_count(static_cast<_CUDA_VSTD::uint16_t>(__init_val));
    }
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE _CUDA_VSTD::uint64_t __arrive_rtn(_CUDA_VSTD::uint32_t __update) const
  {
    _CUDA_VSTD::uint64_t __old = __builtin_amdgcn_ds_atomic_barrier_arrive_rtn_b64(
      reinterpret_cast<__attribute__((address_space(3))) long*>(__lds_word_ptr()), __update);
    llvm_amdgcn_s_wait_dscnt(0);
    return __old;
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE _CUDA_VSTD::uint64_t __query() const
  {
    _CUDA_VSTD::uint64_t& __state_value = const_cast<_CUDA_VSTD::uint64_t&>(value);
    hip::atomic_ref<_CUDA_VSTD::uint64_t, hip::thread_scope_block> __state(__state_value);
    return __state.load(hip::std::memory_order_relaxed);
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE _CUDA_VSTD::uint32_t __pending_count() const
  {
    return __pending_count_from_value(__query());
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE void __add_pending_count(_CUDA_VSTD::uint64_t __count)
  {
    __lds_barrier_t __increment = {};
    __increment.pending_count = __count;
    __scoped_atomic_fetch_add(__lds_word_ptr(), __increment.value, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
    llvm_amdgcn_s_wait_dscnt(0);
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE void __decrement_init_count()
  {
    constexpr _CUDA_VSTD::uint64_t __drop_expected_update = ~((_CUDA_VSTD::uint64_t{1} << 32) - 1);
    __scoped_atomic_fetch_add(__lds_word_ptr(), __drop_expected_update, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
    llvm_amdgcn_s_wait_dscnt(0);
  }

  _CCCL_FORCEINLINE _CCCL_DEVICE void __async_arrive()
  {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_ds_atomic_async_barrier_arrive_b64(
      reinterpret_cast<__attribute__((address_space(3))) long*>(__lds_word_ptr()));
  }

#endif // _CUDA___BARRIER_HIP_HAS_LDS_PHASE_OBJECT
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_LDS_BARRIER_H
