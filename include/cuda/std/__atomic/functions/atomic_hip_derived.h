//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

// NOTE(HIP/AMD): HIP-only counterpart of
// <cuda/std/__atomic/functions/cuda_ptx_derived.h> (AMDGPU rejects the
// NVPTX inline-asm constraints, so this file uses __hip_atomic_* /
// __atomic_* builtins instead). Shape mirrors the upstream file 1:1
// (template-param naming, attribute spellings, overload set,
// __atomic_fetch_update_cuda + __cccl_atomic_op_bind pattern) so
// future upstream cherry-picks translate near-mechanically.
// Intentional divergences:
//   * fetch_{min,max} keep their per-op CAS-loop body for an
//     only-CAS-when-changing optimization upstream's pattern doesn't
//     preserve (inline NOTE at those functions).
//   * __atomic_compare_exchange_n_cuda is HIP-only (the PTX path
//     doesn't need this helper).

#include <hip/hip_runtime.h>
// NOTE(HIP/AMD): pulled in for the std::{plus,minus,bit_and,bit_or,
// bit_xor} functors that the __cccl_atomic_op_bind adapter binds
// to. Matches the upstream include set in cuda_ptx_derived.h.
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/enable_if.h>
// NOTE(HIP/AMD): is_scalar.h transitively pulls in is_pointer.h,
// is_arithmetic.h (and from there is_integral.h + is_floating_point.h).
// Mirrors the upstream CUDA-side transitive chain via
// cuda_ptx_generated_helper.h, which the consumer files
// <cuda/std/__atomic/api/{owned,reference}.h> rely on without including
// the trait headers themselves. Keeping owned.h / reference.h
// byte-identical to upstream by providing the chain on the HIP path.
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
// NOTE(HIP/AMD): Use cuda/std/cstdint for both regular and HIPRTC builds.
// This now works correctly under HIPRTC thanks to the fixes for issue #104
// (using compiler __INT*_TYPE__ builtins and _STDINT_H guards to coexist
// with system headers).
#include <cuda/std/cstdint>

// NOTE(HIP/AMD): pulls in LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH,
// used below to decide whether the native __hip_atomic_fetch_{add,and,or,
// xor,max,min} intrinsics can be used directly, or must instead go through
// the CAS-loop fallback (see the NOTE at those overloads).
#include <libhipcxx/__amd/amd_utils.h>

// NOTE(HIP/AMD): boolean form of LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH
// for use inside enable_if expressions below (the source macro is an
// #ifdef-style presence flag, not a 0/1 value, and `defined(...)` cannot
// appear directly in a template argument).
#if defined(LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH)
#  define _LIBCUDACXX_HIP_ATOMIC_FETCH_NATIVE_SAFE 1
#else
#  define _LIBCUDACXX_HIP_ATOMIC_FETCH_NATIVE_SAFE 0
#endif

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template<typename _Tp, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp) <= 2, int>::type = 0>
[[nodiscard]] bool _CCCL_DEVICE __atomic_compare_exchange_cuda(_Tp volatile *__ptr, _Tp *__expected, const _Tp __desired, bool, int __success_memorder, int __failure_memorder, _Sco __s) noexcept {

    // sizeof<=2 CAS-loop specialisation: emulate sub-word CAS on top
    // of a wider 32-bit CAS by aligning the pointer down, masking the
    // sub-word, and retrying until the unrelated sub-word bits we
    // observed match. Uses the cuda::std::-qualified typedefs from
    // <cuda/std/cstdint>; consistent with the rest of the cuda::std
    // namespace.
    auto const __aligned = (::cuda::std::uint32_t*)((::cuda::std::intptr_t)__ptr & ~(sizeof(::cuda::std::uint32_t) - 1));
    auto const __offset = ::cuda::std::uint32_t((::cuda::std::intptr_t)__ptr & (sizeof(::cuda::std::uint32_t) - 1)) * 8;
    auto const __mask = ((1 << sizeof(_Tp)*8) - 1) << __offset;

    ::cuda::std::uint32_t __old = *__expected << __offset;
    ::cuda::std::uint32_t __old_value;
    while (1) {
        __old_value = (__old & __mask) >> __offset;
        if (__old_value != *__expected)
            break;
        ::cuda::std::uint32_t const __attempt = (__old & ~__mask) | (__desired << __offset);
        if (__atomic_compare_exchange_cuda(__aligned, &__old, &__attempt, true, __success_memorder, __failure_memorder, __s))
            return true;
    }
    *__expected = __old_value;
    return false;
}

template<typename _Tp, typename _Sco>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_load_n_cuda(const _Tp *__ptr, int __memorder, _Sco __s) noexcept {
    _Tp __ret;
    __atomic_load_cuda(__ptr, __ret, __memorder, __s);
    return __ret;
}
template<typename _Tp, typename _Sco>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_load_n_cuda(const _Tp volatile *__ptr, int __memorder, _Sco __s) noexcept {
    _Tp __ret;
    __atomic_load_cuda(__ptr, __ret, __memorder, __s);
    return __ret;
}

template<typename _Tp, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2, int>::type = 0>
void _CCCL_DEVICE __atomic_exchange_cuda(_Tp* __ptr, _Tp& __old, _Tp __new, int __memorder, _Sco __s) noexcept {

    _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    while(!__atomic_compare_exchange_cuda(__ptr, &__expected, __new, true, __memorder, __memorder, __s))
        ;
    __old = __expected;
}

// Functor adapter: binds the second operand of a binary op
// (cuda::std::plus / minus / bit_and / bit_or / bit_xor) so that
// the generic __atomic_fetch_update_cuda CAS loop below can invoke
// it as `__op(__old)` and get back `__old OP __val`. One-to-one
// mirror of upstream's `__cuda_atomic_op_bind` in
// <cuda_ptx_derived.h>.
template <typename _Tp, template <typename> class _Op>
struct __cccl_atomic_op_bind {
    _Tp __val;
    [[nodiscard]] _Tp _CCCL_HOST_DEVICE operator()(_Tp __old) const noexcept {
        return _Op<_Tp>{}(__old, __val);
    }
};

// Generic CAS-loop helper: load, apply the functor to the loaded
// value, CAS the result; on CAS failure re-apply the functor to the
// freshly observed value and retry. Replaces the seven per-op
// hand-coded CAS loops (sizeof<=2 path) for the standard fetch_*
// ops. Mirror of upstream's `__atomic_fetch_update_cuda` in
// <cuda_ptx_derived.h>.
template <typename _Tp, typename _Fn, typename _Sco>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_update_cuda(_Tp volatile* __ptr, const _Fn& __op, int __memorder, _Sco __s) noexcept {
    _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Tp __desired = __op(__expected);
    while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, __s)) {
        __desired = __op(__expected);
    }
    return __expected;
}

// NOTE(HIP/AMD): the type condition includes is_floating_point (not just
// sizeof<=2) UNLESS _LIBCUDACXX_HIP_ATOMIC_FETCH_NATIVE_SAFE. The native
// __hip_atomic_fetch_add lowers `double`/`float` to the hardware FP64/FP32
// atomic (global_atomic_add_f64/f32), which is SILENTLY DROPPED on managed
// (hipMallocManaged, fine-grained-by-default) memory on gfx908 (MI100) and
// gfx90a (MI200) -- e.g. a hash-groupby SUM/MEAN of doubles returns 0. Per
// AMD's GPU atomics support tables, this is fixed from CDNA3 onward
// (gfx940/941/942/950 -- see LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH
// in <libhipcxx/__amd/amd_utils.h>), where the native instruction is correct
// on managed memory too. So: route floating-point add/sub through the CAS
// loop (__atomic_fetch_update_cuda), exactly as fetch_max/fetch_min already
// do for FP, only on architectures where the native op is unsafe; on
// CDNA3+ let floating-point add/sub fall through to the native
// __hip_atomic_fetch_add path (see atomic_hip_generated.h) for full
// performance.
template<typename _Tp, typename _Up, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2 || (::cuda::std::is_floating_point<_Tp>::value && !_LIBCUDACXX_HIP_ATOMIC_FETCH_NATIVE_SAFE), int>::type = 0>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_add_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::plus>{static_cast<_Tp>(__val)}, __memorder, __s);
}

template<typename _Tp, typename _Up, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2 || (::cuda::std::is_floating_point<_Tp>::value && !_LIBCUDACXX_HIP_ATOMIC_FETCH_NATIVE_SAFE), int>::type = 0>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_sub_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::minus>{static_cast<_Tp>(__val)}, __memorder, __s);
}

// NOTE(HIP/AMD): unlike fetch_add/fetch_sub, the native __hip_atomic_fetch_*
// used here has no fine-grained-memory-safe case at all sizes for
// fetch_and/or/xor -- probing on real gfx90a hardware showed the native op
// silently drops on managed memory regardless of operand width (verified for
// both 8-bit and 32-bit operands). So route through the CAS loop
// unconditionally on architectures where LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH
// is unset; on CDNA3+ fall through to the native path (see
// atomic_hip_generated.h) for full performance.
//
// This split is a plain '#if', not an enable_if, because the condition does
// not depend on _Tp at all -- an enable_if that is always-false for every
// possible _Tp is rejected by Clang ("enable_if cannot be used to disable
// this declaration"), since such a template could never be instantiated.
#if !_LIBCUDACXX_HIP_ATOMIC_FETCH_NATIVE_SAFE
template<typename _Tp, typename _Up, typename _Sco>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_and_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::bit_and>{static_cast<_Tp>(__val)}, __memorder, __s);
}

template<typename _Tp, typename _Up, typename _Sco>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_or_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::bit_or>{static_cast<_Tp>(__val)}, __memorder, __s);
}

template<typename _Tp, typename _Up, typename _Sco>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_xor_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::bit_xor>{static_cast<_Tp>(__val)}, __memorder, __s);
}
#endif // !_LIBCUDACXX_HIP_ATOMIC_FETCH_NATIVE_SAFE

// NOTE(HIP/AMD): fetch_min / fetch_max keep their per-op CAS-loop
// implementation rather than going through __atomic_fetch_update_cuda
// because they carry a load-bearing OPTIMIZATION: the loop only
// CASes when the proposed value would actually change the stored
// value (the `while(__desired == __val && ...)` guard). For a hot
// max/min loop converging on a stable best value, this avoids a
// CAS per iteration once the stored value is already <= / >= the
// proposed value. The generic adapter would always CAS, even when
// no change is needed, so folding these in would be a (small but
// measurable) perf regression on min/max heavy workloads.
//
// Like fetch_and/or/xor above, the native __hip_atomic_fetch_max/min is
// silently dropped on managed (fine-grained) memory regardless of operand
// type or width on architectures where LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH
// is unset (probed on real gfx90a hardware for both int32 and uint8) -- so
// this CAS-loop is used unconditionally there, for every type, not just
// sizeof<=2 / floating-point. On CDNA3+ fall through to the native path
// (see atomic_hip_generated.h) for full performance.
//
// Plain '#if' (see fetch_and/or/xor above) rather than enable_if, since the
// condition is not _Tp-dependent.
#if !_LIBCUDACXX_HIP_ATOMIC_FETCH_NATIVE_SAFE
template<typename _Tp, typename _Up, typename _Sco>
[[nodiscard]] _Tp _CCCL_HOST_DEVICE __atomic_fetch_max_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Tp __desired = __expected > __val ? __expected : __val;

    while(__desired == __val &&
            !__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, __s)) {
        __desired = __expected > __val ? __expected : __val;
    }

    return __expected;
}

template<typename _Tp, typename _Up, typename _Sco>
[[nodiscard]] _Tp _CCCL_HOST_DEVICE __atomic_fetch_min_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Tp __desired = __expected < __val ? __expected : __val;

    while(__desired == __val &&
            !__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, __s)) {
        __desired = __expected < __val ? __expected : __val;
    }

    return __expected;
}
#endif // !_LIBCUDACXX_HIP_ATOMIC_FETCH_NATIVE_SAFE

template<typename _Tp, typename _Sco>
void _CCCL_DEVICE __atomic_store_n_cuda(_Tp *__ptr, _Tp __val, int __memorder, _Sco __s) noexcept {
    __atomic_store_cuda(__ptr, __val, __memorder, __s);
}
template<typename _Tp, typename _Sco>
void _CCCL_DEVICE __atomic_store_n_cuda(_Tp volatile *__ptr, _Tp __val, int __memorder, _Sco __s) noexcept {
    __atomic_store_cuda(__ptr, __val, __memorder, __s);
}

template<typename _Tp, typename _Sco>
[[nodiscard]] bool _CCCL_DEVICE __atomic_compare_exchange_n_cuda(_Tp volatile *__ptr, _Tp *__expected, _Tp __desired, bool __weak, int __success_memorder, int __failure_memorder, _Sco __s) noexcept {
    return __atomic_compare_exchange_cuda(__ptr, __expected, __desired, __weak, __success_memorder, __failure_memorder, __s);
}

template<typename _Tp, typename _Sco>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_exchange_n_cuda(_Tp volatile * __ptr, _Tp __val, int __memorder, _Sco __s) noexcept {
    _Tp __ret;
    __atomic_exchange_cuda(__ptr, __ret, __val, __memorder, __s);
    return __ret;
}

template<typename _Tp, typename _Sco>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_exchange_n_cuda(_Tp * __ptr, _Tp __val, int __memorder, _Sco __s) noexcept {
    _Tp __ret;
    __atomic_exchange_cuda(__ptr, __ret, __val, __memorder, __s);
    return __ret;
}

_CCCL_DEVICE static inline void __atomic_signal_fence_cuda(int) noexcept {
    asm volatile("":::"memory");
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>
