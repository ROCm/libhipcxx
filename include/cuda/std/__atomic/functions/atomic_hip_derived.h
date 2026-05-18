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

// NOTE(HIP/AMD): file divergence from upstream. This file is the
// HIP-only counterpart of upstream's
//   <cuda/std/__atomic/functions/cuda_ptx_derived.h>
// (which uses NVPTX inline asm; AMDGPU rejects the NVPTX-specific
// inline-asm constraints, so we provide a parallel implementation
// built on HIP's __hip_atomic_* / __atomic_* builtins instead).
//
// Because the two files have different paths, upstream PRs that
// modernise cuda_ptx_derived.h do NOT auto-apply to this file via
// cherry-pick.
//
// Migration to the upstream shape is being done in phases (PR #217
// thread on atomic_hip_derived.h:77). Phase status:
//
//   [x] Phase 1: rename `_Type/_Scope/_Delta` -> `_Tp/_Sco/_Up`;
//                add [[nodiscard]] to value-returning helpers; add
//                noexcept everywhere. ZERO RISK -- pure annotations.
//   [x] Phase 2: switch bare `__device__` / `__host__ __device__`
//                spellings to the CCCL portable macros `_CCCL_DEVICE`
//                / `_CCCL_HOST_DEVICE`. Pure macro rename; the
//                expanded text is identical. (A future Phase 2.5
//                may promote `_CCCL_DEVICE` to `_CCCL_DEVICE_API`
//                where upstream uses the API spelling, picking up
//                the `_CCCL_VISIBILITY_HIDDEN` +
//                `_CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION` markers
//                from the API-suffixed macro.)
//   [x] Phase 3: add the non-volatile overloads of load_n and
//                store_n that upstream provides. Pure additions;
//                existing callers continue to bind to the volatile
//                overload via implicit non-vol -> vol qualification.
//                compare_exchange_n is HIP-specific (not in upstream)
//                and stays single-overload by intent.
//   [x] Phase 4: replace per-op CAS loops with a single generic
//                `__atomic_fetch_update_cuda<_Tp, _Fn>` + a
//                `__cccl_atomic_op_bind<_Tp, _Op>` adapter (matches
//                upstream's pattern). Five ops (add/sub/and/or/xor)
//                are folded into the generic helper; fetch_{min,max}
//                keep their per-op CAS-loop because of their
//                load-bearing 'only-CAS-when-changing' optimization
//                that the upstream pattern does not preserve.
//   [ ] Phase 5: wrap the file body in _CCCL_BEGIN_NAMESPACE_CUDA_STD
//                and update consumer references in
//                cuda/std/__atomic/types/base.h. MEDIUM RISK --
//                ABI surface change.
//   [ ] Phase 6: cleanup; retire this NOTE block.
//
// Behaviour is correct at every phase boundary; the divergence is
// stylistic, not semantic.

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
using ::intptr_t;
using ::uint32_t;
template<typename _Tp, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp) <= 2, int>::type = 0>
[[nodiscard]] bool _CCCL_DEVICE __atomic_compare_exchange_cuda(_Tp volatile *__ptr, _Tp *__expected, const _Tp __desired, bool, int __success_memorder, int __failure_memorder, _Sco __s) noexcept {

    auto const __aligned = (uint32_t*)((intptr_t)__ptr & ~(sizeof(uint32_t) - 1));
    auto const __offset = uint32_t((intptr_t)__ptr & (sizeof(uint32_t) - 1)) * 8;
    auto const __mask = ((1 << sizeof(_Tp)*8) - 1) << __offset;

    uint32_t __old = *__expected << __offset;
    uint32_t __old_value;
    while (1) {
        __old_value = (__old & __mask) >> __offset;
        if (__old_value != *__expected)
            break;
        uint32_t const __attempt = (__old & ~__mask) | (__desired << __offset);
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

template<typename _Tp, typename _Up, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2, int>::type = 0>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_add_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::plus>{static_cast<_Tp>(__val)}, __memorder, __s);
}

template<typename _Tp, typename _Up, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2, int>::type = 0>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_sub_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::minus>{static_cast<_Tp>(__val)}, __memorder, __s);
}

template<typename _Tp, typename _Up, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2, int>::type = 0>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_and_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::bit_and>{static_cast<_Tp>(__val)}, __memorder, __s);
}

template<typename _Tp, typename _Up, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2, int>::type = 0>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_or_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::bit_or>{static_cast<_Tp>(__val)}, __memorder, __s);
}

template<typename _Tp, typename _Up, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2, int>::type = 0>
[[nodiscard]] _Tp _CCCL_DEVICE __atomic_fetch_xor_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    return __atomic_fetch_update_cuda(__ptr, __cccl_atomic_op_bind<_Tp, ::cuda::std::bit_xor>{static_cast<_Tp>(__val)}, __memorder, __s);
}

// NOTE(HIP/AMD): fetch_min / fetch_max keep their per-op CAS-loop
// implementation rather than going through __atomic_fetch_update_cuda
// because they carry a load-bearing OPTIMIZATION: the loop only
// CASes when the proposed value would actually change the stored
// value (the `while(__desired == __val && ...)` guard). For a hot
// max/min loop converging on a stable best value, this avoids a
// CAS per iteration once the stored value is already <= / >= the
// proposed value. The generic adapter would always CAS, even when
// no change is needed, so folding these in would be a (small but
// measurable) perf regression on min/max heavy workloads. Upstream
// cuda_ptx_derived.h doesn't carry this optimization for its
// generic adapter path, but their underlying PTX `atom.min/max`
// instruction is hardware-fast for the common arithmetic types
// (sizeof >= 4); HIP only hits the CAS-loop fallback for the
// sizeof<=2 and floating-point arms, where the optimization
// matters more.
template<typename _Tp, typename _Up, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2 || ::cuda::std::is_floating_point<_Tp>::value, int>::type = 0>
[[nodiscard]] _Tp _CCCL_HOST_DEVICE __atomic_fetch_max_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Tp __desired = __expected > __val ? __expected : __val;

    while(__desired == __val &&
            !__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, __s)) {
        __desired = __expected > __val ? __expected : __val;
    }

    return __expected;
}

template<typename _Tp, typename _Up, typename _Sco, typename ::cuda::std::enable_if<sizeof(_Tp)<=2 || ::cuda::std::is_floating_point<_Tp>::value, int>::type = 0>
[[nodiscard]] _Tp _CCCL_HOST_DEVICE __atomic_fetch_min_cuda(_Tp volatile *__ptr, _Up __val, int __memorder, _Sco __s) noexcept {
    _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Tp __desired = __expected < __val ? __expected : __val;

    while(__desired == __val &&
            !__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, __s)) {
        __desired = __expected < __val ? __expected : __val;
    }

    return __expected;
}

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

static inline _CCCL_DEVICE void __atomic_signal_fence_cuda(int) noexcept {
    asm volatile("":::"memory");
}
