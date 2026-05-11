//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#ifndef HIP_INTRINSICS_H
#define HIP_INTRINSICS_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include <fmt/format.h>

// =============================================================================
// HIP-side __atomic_*_cuda emitter.
//
// Emits the ENTIRE 'atomic_hip_generated.h' header (header + scope blocks +
// footer) when 'codegen --hip <out>' is invoked.
//
// Scope coverage (Tier 2):
//   * 'block' / 'device' / 'system' scope blocks are emitted by
//     'FormatHipScope' -- ~12-13 templated overloads per scope, routed
//     through clang's '__hip_atomic_*' builtins with a baked-in
//     '__HIP_MEMORY_SCOPE_*' macro. The (size, type) cross-product NV
//     enumerates explicitly is handled implicitly here -- clang HIP
//     builtins are runtime-typed and runtime-ordered.
//   * 'cluster' scope block is emitted by 'FormatHipScopeUnsupported'.
//     The signatures match the supported scopes (so consumers who do
//     'cuda::atomic_ref<T, thread_scope_cluster>' overload-resolve to
//     a real overload), but every body is just a 'static_assert'
//     fenced behind '__always_false_v<_Type>' so it only fires at
//     instantiation. Reasoning: AMDGCN has no cluster-scope equivalent.
//     Mapping cluster -> AGENT (next-larger scope) silently
//     over-synchronises and lies to the consumer about what was
//     actually compiled; mapping cluster -> WORKGROUP under-synchronises.
//     A clean static_assert at instantiation is the honest answer.
//
// Type coverage (Tier 2):
//   * Per-template '_Type' is sized at instantiation. For
//     'sizeof(_Type) > 8', the load/store/compare_exchange/exchange
//     overloads emit a 'static_assert(__always_false_v<_Type>, ...)'
//     inside an 'if constexpr' branch -- clang's '__hip_atomic_*'
//     builtins cap at 64-bit, so any wider type (incl. __int128, 128-bit
//     structs, long double on some ABIs) would otherwise hit a noisier
//     clang diagnostic somewhere deeper. The if-constexpr keeps the
//     function signature singular and keeps the supported-size path
//     identical to Tier 1's emit.
//   * fetch_{add,sub,and,or,xor,max,min} are NOT gated on size in the
//     same way -- the existing atomic_hip_derived.h CAS-loop fallback
//     path supplies overloads for sub-word and oversized types.
//
// Order / semantic / MMIO axes (Tier 2):
//   * Order:  passed through as 'int __memorder' to the clang builtin
//     directly; no compile-time semantic tag layer needed (PTX needs
//     that because the asm string has to know the order at compile
//     time; clang HIP builtins do not).
//   * Volatile: collapsed to relaxed by the clang builtin's runtime
//     order semantics; no separate emission.
//   * MMIO:   not emitted on HIP. AMDGCN has no per-op MMIO modifier;
//     the equivalent is a one-time allocation-side cache attribute via
//     'hipHostMalloc(..., hipHostMallocCoherent)'. A consumer that
//     ever instantiates the NV-side MMIO overload via the public
//     atomic_ref API surface gets a "no matching overload" template
//     error pointing at the '__atomic_cuda_mmio_enable' parameter --
//     honest about the gap.
//
// Architectural shape (Tier 2 picks 'flat'):
//   * Stay with Tier 1's 'public API only' shape. Don't introduce the
//     NV-side '__cuda_atomic_*' + bind-helper + memory-order dispatch
//     layering. clang HIP builtins consume runtime 'int memorder'
//     directly, so the NV layering would be pure overhead with no
//     upstream-tracking benefit on HIP (the layer-4 intrinsics are
//     necessarily backend-specific either way).
//
// Per-scope quirks preserved verbatim from Tier 1 (so the Tier 1 -> Tier 2
// in-tree diff is small enough to eyeball):
//   - has_ptr_fetch_sub  : false for block (no ptr fetch_sub in the
//                          in-tree file's block-scope section); true for
//                          device/system.
//   - cas_branches_on_weak : false for block/device (anonymous bool param,
//                            always weak CAS); true for system (named
//                            __is_weak param, branches between weak and
//                            strong CAS).
// =============================================================================

// Stream-emit the contents of the file pointed to by the LICENSE_FILE env var
// (set by the codegen build target -- see codegen/CMakeLists.txt). The license
// text is intentionally NOT embedded as a raw-string literal here, for two
// reasons:
//
//   1. Avoids a duplicate license block in this file. With the literal in a
//      raw-string, 'hip_intrinsics.h' would carry one license header for its
//      own source code (top of file, lines 1-26) and an identical one inside
//      'FormatHipHeader' for the EMITTED output. License-verification tools
//      that scan files line-by-line would see two blocks and either flag the
//      file as inconsistent or strip the second as a 'duplicate', silently
//      breaking the emitted header. Reading the license at codegen time keeps
//      each file with at most one license block.
//
//   2. Single source of truth for the emitted license text. The license lives
//      at 'codegen/generators/atomic_hip_generated.license.txt' as data; both
//      the on-disk file and 'atomic_hip_generated.h' (after regeneration)
//      contain byte-identical text. Updating the license is a one-file edit
//      followed by a regen, instead of touching this template.
//
// Aborts on missing / unreadable LICENSE_FILE so a misconfigured build fails
// loudly instead of silently shipping a license-less generated header.
inline void FormatHipEmitLicense(std::ostream& out)
{
  const char* __license_file = std::getenv("LICENSE_FILE");
  if (__license_file == nullptr || *__license_file == '\0')
  {
    std::cerr << "FormatHipHeader: LICENSE_FILE env var must be set to the path of\n"
              << "  codegen/generators/atomic_hip_generated.license.txt (or equivalent).\n"
              << "  Set via 'cmake -E env LICENSE_FILE=... codegen --hip ...' (already\n"
              << "  wired in codegen/CMakeLists.txt's libcudacxx.atomics.codegen.hip target).\n";
    std::exit(2);
  }
  std::ifstream __license_stream(__license_file);
  if (!__license_stream.is_open())
  {
    std::cerr << "FormatHipHeader: cannot open LICENSE_FILE='" << __license_file << "' for reading\n";
    std::exit(2);
  }
  out << __license_stream.rdbuf();
}

inline void FormatHipHeader(std::ostream& out)
{
  FormatHipEmitLicense(out);
  out << R"XXX(
// This is an autogenerated file. Edits will be lost on the next regeneration --
// modify codegen/generators/hip_intrinsics.h instead and run
//   cmake --build <build> --target libcudacxx.atomics.codegen.hip.install
// 'libcudacxx.test.atomics.codegen.hip.diff' guards against in-tree drift.
// clang-format off

#pragma once

#include <hip/hip_runtime.h>

//#include <cuda/std/cassert>
//#include <cuda/std/cstdint>

#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>

#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__atomic/functions/cuda_local.h>
//#include <cuda/std/__atomic/order.h>
//#include <cuda/std/__atomic/functions/common.h>
//#include <cuda/std/__atomic/functions/cuda_ptx_generated_helper.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD
)XXX";
}

inline void FormatHipFooter(std::ostream& out)
{
  out << R"XXX(
_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>
)XXX";
}

inline void FormatHipScope(std::ostream& out,
                           const std::string& scope_tag,
                           const std::string& fence_func,
                           const std::string& scope_macro,
                           bool has_ptr_fetch_sub,
                           bool cas_branches_on_weak)
{
  // thread_fence
  out << fmt::format(R"XXX(
static inline __device__ void __atomic_thread_fence_cuda(int __memorder, {0}) {{
    {1}();
}}
)XXX",
                     scope_tag, fence_func);

  // load -- gated on sizeof(_Type) > 8 for clang HIP builtin's 64-bit cap.
  out << fmt::format(R"XXX(
template<class _Type>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type& __dst, int __memorder, {0}) {{
    if constexpr (sizeof(_Type) > 8) {{
        static_assert(::cuda::std::__always_false_v<_Type>,
                      ">64-bit atomic load is not supported on HIP (clang HIP __hip_atomic_load caps at 64-bit)");
    }} else {{
        if (__cuda_load_weak_if_local(__ptr, &__dst, sizeof(_Type))) return;
        __dst = __hip_atomic_load(__ptr, __memorder, {1});
    }}
}}
)XXX",
                     scope_tag, scope_macro);

  // store -- gated on sizeof(_Type) > 8 (clang HIP __hip_atomic_store cap).
  out << fmt::format(R"XXX(
template<class _Type>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type& __val, int __memorder, {0}) {{
    if constexpr (sizeof(_Type) > 8) {{
        static_assert(::cuda::std::__always_false_v<_Type>,
                      ">64-bit atomic store is not supported on HIP (clang HIP __hip_atomic_store caps at 64-bit)");
    }} else {{
        if (__cuda_store_weak_if_local(__ptr, &__val, sizeof(_Type))) return;
        __hip_atomic_store(__ptr, __val, __memorder, {1});
    }}
}}
)XXX",
                     scope_tag, scope_macro);

  // compare_exchange -- gated on sizeof(_Type) > 8 (clang HIP cap).
  if (cas_branches_on_weak)
  {
    out << fmt::format(R"XXX(
template<class _Type>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type __desired, bool __is_weak, int __success_memorder, int __failure_memorder, {0}) {{
    if constexpr (sizeof(_Type) > 8) {{
        static_assert(::cuda::std::__always_false_v<_Type>,
                      ">64-bit atomic compare_exchange is not supported on HIP (clang HIP __hip_atomic_compare_exchange_* caps at 64-bit)");
        return false;
    }} else {{
        bool __success;
        if (__cuda_compare_exchange_weak_if_local(__ptr, __expected, &__desired, &__success)) return __success;
        if(__is_weak)
            return __hip_atomic_compare_exchange_weak(__ptr, __expected, __desired, __success_memorder, __failure_memorder, {1});
        else
            return __hip_atomic_compare_exchange_strong(__ptr, __expected, __desired, __success_memorder, __failure_memorder, {1});
    }}
}}
)XXX",
                       scope_tag, scope_macro);
  }
  else
  {
    out << fmt::format(R"XXX(
template<class _Type>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type __desired, bool, int __success_memorder, int __failure_memorder, {0}) {{
    if constexpr (sizeof(_Type) > 8) {{
        static_assert(::cuda::std::__always_false_v<_Type>,
                      ">64-bit atomic compare_exchange is not supported on HIP (clang HIP __hip_atomic_compare_exchange_weak caps at 64-bit)");
        return false;
    }} else {{
        bool __success;
        if (__cuda_compare_exchange_weak_if_local(__ptr, __expected, &__desired, &__success)) return __success;
        return __hip_atomic_compare_exchange_weak(__ptr, __expected, __desired, __success_memorder, __failure_memorder, {1});
    }}
}}
)XXX",
                       scope_tag, scope_macro);
  }

  // exchange -- gated on sizeof(_Type) > 8 (clang HIP __hip_atomic_exchange cap).
  out << fmt::format(R"XXX(
template<class _Type>
__device__ void __atomic_exchange_cuda(volatile _Type* __ptr, _Type& __old, _Type __new, int __memorder, {0}) {{
    if constexpr (sizeof(_Type) > 8) {{
        static_assert(::cuda::std::__always_false_v<_Type>,
                      ">64-bit atomic exchange is not supported on HIP (clang HIP __hip_atomic_exchange caps at 64-bit)");
    }} else {{
        if (__cuda_exchange_weak_if_local(__ptr, &__new, &__old)) return;
        __old = __hip_atomic_exchange(__ptr, __new, __memorder, {1});
    }}
}}
)XXX",
                     scope_tag, scope_macro);

  // fetch_{and, or, xor, add, max, min}: same shape, direct __hip_atomic_fetch_*
  for (const char* op : {"and", "or", "xor", "add", "max", "min"})
  {
    out << fmt::format(R"XXX(
template<class _Type>
__device__ _Type __atomic_fetch_{2}_cuda(volatile _Type *__ptr, _Type __val, int __memorder, {0}) {{
    _Type __ret;
    if (__cuda_fetch_{2}_weak_if_local(__ptr, __val, &__ret)) return __ret;
    return __hip_atomic_fetch_{2}(__ptr, __val, __memorder, {1});
}}
)XXX",
                       scope_tag, scope_macro, op);
  }

  // fetch_sub: implemented as fetch_add with negated value (HIP has no native fetch_sub builtin)
  out << fmt::format(R"XXX(
template<class _Type>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, {0}) {{
    _Type __ret;
    if (__cuda_fetch_sub_weak_if_local(__ptr, __val, &__ret)) return __ret;
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, {1});
}}
)XXX",
                     scope_tag, scope_macro);

  // pointer fetch_add
  out << fmt::format(R"XXX(
template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, {0}) {{
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, {1});
}}
)XXX",
                     scope_tag, scope_macro);

  // pointer fetch_sub: only for device/system (block scope omits it in the in-tree
  // file; preserve to keep this Tier-1 emit byte-equivalent. Note: also no blank
  // line between ptr fetch_add and ptr fetch_sub -- mirrors the in-tree shape.)
  if (has_ptr_fetch_sub)
  {
    out << fmt::format(R"XXX(template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, {0}) {{
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, {1});
}}
)XXX",
                       scope_tag, scope_macro);
  }
}

// Emit a scope block whose every overload is a static_assert at instantiation.
// Used for HIP-unsupported scope tags (today: 'cluster' -- AMDGCN has no
// cluster-scope equivalent and any silent mapping would lie about what was
// actually compiled). The signatures match the supported scopes 1:1 so that
// overload resolution from the public 'cuda::atomic_ref<T, scope>' API picks
// up THESE overloads on cluster scope and produces the static_assert message
// at the right call site, instead of a "no matching function" template-tower.
//
// Body shape: 'static_assert(__always_false_v<_Type>, "<msg>")' is
// value-dependent on the template parameter, so it is NOT evaluated at
// template definition time -- only when the function is instantiated. That
// keeps the file compilable in NV-mode builds where these overloads exist
// but are never instantiated.
//
// Where the function has a non-void return type we emit a trivially-typed
// fallback return ('return false;' / 'return _Type{};' / 'return nullptr;')
// after the static_assert. The fallback is unreachable at runtime (the
// static_assert aborts compilation before any caller is generated) but
// keeps the function well-formed for the tooling that walks the file
// looking for return-statement coverage.
inline void FormatHipScopeUnsupported(std::ostream& out, const std::string& scope_tag, const std::string& reason)
{
  // thread_fence (void). The supported scopes' '__atomic_thread_fence_cuda'
  // is a NON-template function. Mirroring that shape here verbatim would make
  // the static_assert fire at PARSE time -- there is no value-dependent
  // expression in the body to defer it to instantiation. Wrap as a function
  // template with a dummy template parameter '_Dummy' that the static_assert
  // references via '__always_false_v<_Dummy>' so the assertion is
  // value-dependent on '_Dummy' and only fires at instantiation. Overload
  // resolution still picks this for any call shape
  // '(int, __thread_scope_cluster_tag)' because _Dummy has a default.
  out << fmt::format(R"XXX(
template <class _Dummy = void>
static inline __device__ void __atomic_thread_fence_cuda(int, {0}) {{
    static_assert(::cuda::std::__always_false_v<_Dummy>, "{1}");
}}
)XXX",
                     scope_tag, reason);

  // load (void)
  out << fmt::format(R"XXX(
template<class _Type>
__device__ void __atomic_load_cuda(const volatile _Type *, _Type&, int, {0}) {{
    static_assert(::cuda::std::__always_false_v<_Type>, "{1}");
}}
)XXX",
                     scope_tag, reason);

  // store (void)
  out << fmt::format(R"XXX(
template<class _Type>
__device__ void __atomic_store_cuda(volatile _Type *, _Type&, int, {0}) {{
    static_assert(::cuda::std::__always_false_v<_Type>, "{1}");
}}
)XXX",
                     scope_tag, reason);

  // compare_exchange (bool)
  out << fmt::format(R"XXX(
template<class _Type>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *, _Type *, const _Type, bool, int, int, {0}) {{
    static_assert(::cuda::std::__always_false_v<_Type>, "{1}");
    return false;
}}
)XXX",
                     scope_tag, reason);

  // exchange (void)
  out << fmt::format(R"XXX(
template<class _Type>
__device__ void __atomic_exchange_cuda(volatile _Type*, _Type&, _Type, int, {0}) {{
    static_assert(::cuda::std::__always_false_v<_Type>, "{1}");
}}
)XXX",
                     scope_tag, reason);

  // fetch_{and, or, xor, add, max, min, sub} (_Type)
  for (const char* op : {"and", "or", "xor", "add", "max", "min", "sub"})
  {
    out << fmt::format(R"XXX(
template<class _Type>
__device__ _Type __atomic_fetch_{2}_cuda(volatile _Type *, _Type, int, {0}) {{
    static_assert(::cuda::std::__always_false_v<_Type>, "{1}");
    return _Type{{}};
}}
)XXX",
                       scope_tag, reason, op);
  }

  // pointer fetch_add / fetch_sub (_Type*) -- emit both for the cluster path
  // even though Tier 1's 'block' scope omits ptr fetch_sub (the reason for
  // that quirk -- preserving the in-tree file -- doesn't apply here, so we
  // give the unsupported-scope version full coverage).
  out << fmt::format(R"XXX(
template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *, ptrdiff_t, int, {0}) {{
    static_assert(::cuda::std::__always_false_v<_Type>, "{1}");
    return nullptr;
}}
template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *, ptrdiff_t, int, {0}) {{
    static_assert(::cuda::std::__always_false_v<_Type>, "{1}");
    return nullptr;
}}
)XXX",
                     scope_tag, reason);
}

inline void FormatHip(std::ostream& out)
{
  FormatHipHeader(out);
  FormatHipScope(out,
                 /*scope_tag=*/"__thread_scope_block_tag",
                 /*fence_func=*/"__threadfence_block",
                 /*scope_macro=*/"__HIP_MEMORY_SCOPE_WORKGROUP",
                 /*has_ptr_fetch_sub=*/false,
                 /*cas_branches_on_weak=*/false);
  // Cluster scope is HIP-unsupported (AMDGCN has no equivalent). Emit a
  // signature-matching block that static_asserts at instantiation, BEFORE
  // the device/system scopes -- ordering mirrors the upstream NV file which
  // emits cluster between block and device.
  FormatHipScopeUnsupported(out,
                            /*scope_tag=*/"__thread_scope_cluster_tag",
                            /*reason=*/"thread_scope_cluster is not supported on HIP "
                                       "(AMDGCN has no cluster-scope equivalent)");
  FormatHipScope(out,
                 /*scope_tag=*/"__thread_scope_device_tag",
                 /*fence_func=*/"__threadfence",
                 /*scope_macro=*/"__HIP_MEMORY_SCOPE_AGENT",
                 /*has_ptr_fetch_sub=*/true,
                 /*cas_branches_on_weak=*/false);
  FormatHipScope(out,
                 /*scope_tag=*/"__thread_scope_system_tag",
                 /*fence_func=*/"__threadfence_system",
                 /*scope_macro=*/"__HIP_MEMORY_SCOPE_SYSTEM",
                 /*has_ptr_fetch_sub=*/true,
                 /*cas_branches_on_weak=*/true);
  FormatHipFooter(out);
}

#endif // HIP_INTRINSICS_H
