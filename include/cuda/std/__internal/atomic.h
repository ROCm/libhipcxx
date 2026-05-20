//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

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

#ifndef _CUDA_STD___INTERNAL_ATOMIC_H
#define _CUDA_STD___INTERNAL_ATOMIC_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__internal/features.h>

#if _CCCL_HAS_CUDA_COMPILER()
#  define _CCCL_ATOMIC_ALWAYS_LOCK_FREE(size, ptr) (size <= 8)
#elif _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(GCC)
#  define _CCCL_ATOMIC_ALWAYS_LOCK_FREE(...) __atomic_always_lock_free(__VA_ARGS__)
#endif // _CCCL_CUDA_COMPILER

// Enable bypassing automatic storage checks in atomics when using CTK 12.2 and below and if NDEBUG is defined.
// A compiler bug prevents the safe use of `__is_local` and PTX spacep until after 13.0.
#ifndef _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE
#  if _CCCL_CUDACC_BELOW(13, 1) && !defined(NDEBUG)
#    define _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE
#  endif // _CCCL_CUDACC_BELOW(13, 1)
#endif // _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE

// NOTE(HIP/AMD): the SAFE automatic-storage path inlines a
// '__cuda_is_local(ptr)' check (which lowers to
// '__builtin_amdgcn_is_private') into every cuda::atomic<T> op
// in cuda/std/__atomic/functions/atomic_hip_generated.h. On the
// HIPCC offline-compile path that is fine and is needed for
// atomic.local.pass.cpp (a stack-local cuda::atomic<T> would
// otherwise trap because AMDGCN has no HW atomics on the private
// address space). The HIPRTC runtime-compile pipeline (COMGR ->
// inline-clang -> bitcode link) historically choked on the
// resulting IR with a backend codegen failure ("V_CMP_NE_U32_e32 0,
// $src_shared_base, ..." spam followed by HIPRTC_ERROR_LINKING)
// for any TU that exercised a wide spread of atomic<T>
// instantiations -- atomic_fetch_min / max, address(_ref) /
// constness, and compare_exchange_weak{,_explicit} all regressed.
// Force-enable the UNSAFE knob on the HIPRTC path so
// __cuda_is_local() short-circuits to false and the SAFE shims
// degenerate to a single tail-call into the underlying
// __hip_atomic_* builtin.
//
// The underlying AMDGPU codegen bug is FIXED in ROCm 7.2.3 (hotfix
// on the otherwise-frozen 7.2 line) and in ROCm 7.13.0 (mainline);
// on those releases the SAFE path lowers correctly and the WAR is
// skipped so HIPRTC builds get the same SAFE-path coverage as HIPCC
// offline. atomic.local.pass.cpp's main() body is wrapped in
// '#if !defined(_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE)' so on still-
// broken ROCm the test trivially passes (no UNSUPPORTED marker
// required), and on fixed ROCm it exercises the stack-allocated
// cuda::atomic<T> path for real.
//
// Conservative default: if ROCM_VERSION_* macros are not available
// the WAR stays active. <amd/amd_utils.h> is pulled in only on the
// HIPRTC branch and is itself HIPRTC-safe (it drops the
// <hip/hip_runtime.h> include there and only brings in the
// lightweight <rocm-core/rocm_version.h>).
#if defined(_CCCL_COMPILER_HIPRTC) && !defined(_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE)
#  include <amd/amd_utils.h>
#  if !(LIBHIPCXX_ROCM_VERSION_EQ(7, 2, 3) || LIBHIPCXX_ROCM_VERSION_GE(7, 13, 0))
#    define _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE
#  endif
#endif // _CCCL_COMPILER_HIPRTC && !_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE && ROCm-still-broken

#define _CCCL_ATOMIC_FLAG_TYPE int

// Clang provides 128b atomics as a builtin
#if defined(CCCL_ENABLE_EXPERIMENTAL_HOST_ATOMICS_128B)
#  define _CCCL_HOST_128_ATOMICS_ENABLED() 1
#  define _CCCL_HOST_128_ATOMICS_MAYBE()   0
// GCC does not provide 128b atomics, but they may be available as a library, this requires opt-in usage.
// See: https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html "-mcx16" for more
#elif _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(GCC)
#  define _CCCL_HOST_128_ATOMICS_ENABLED() 0
#  define _CCCL_HOST_128_ATOMICS_MAYBE()   1
#else
#  define _CCCL_HOST_128_ATOMICS_ENABLED() 0
#  define _CCCL_HOST_128_ATOMICS_MAYBE()   0
#endif

#endif // _CUDA_STD___INTERNAL_ATOMIC_H
