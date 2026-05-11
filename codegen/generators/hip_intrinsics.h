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
// Emits the ENTIRE 'atomic_hip_generated.h' header (header + 3 scope blocks +
// footer) when 'codegen --hip <out>' is invoked.
//
// Scope of this emitter (Tier 1, see commit message):
//   * Output is intended to be byte-equivalent to the file that has lived in
//     'include/cuda/std/__atomic/functions/atomic_hip_generated.h' since the
//     HIP backend was first added, modulo a single whitespace fix on the
//     system-scope 'fetch_add' overload (3-space indent normalised to 4).
//   * No new operations, sizes, types, semantics, scopes or MMIO axes are
//     added relative to the hand-maintained file. Each of the three scope
//     blocks below emits exactly the (12 or 13) overloads the in-tree file
//     already defines, parameterised only by:
//       - scope_tag       : __thread_scope_{block,device,system}_tag
//       - fence_func      : __threadfence_{block,,system}
//       - scope_macro     : __HIP_MEMORY_SCOPE_{WORKGROUP,AGENT,SYSTEM}
//       - has_ptr_fetch_sub : false for block (the in-tree file does not
//                             define it for block scope), true for the other
//                             two -- preserved as-is to keep the output
//                             byte-equivalent.
//       - cas_branches_on_weak : false for block/device (the in-tree file
//                                accepts a 'bool' but ignores it and always
//                                does a weak CAS), true for system (in-tree
//                                file branches between weak and strong).
//                                Preserved as-is.
//
// Tier 2 (a follow-up commit) will replace these scope-only loops with the
// per-(size, type, semantic, scope, mmio) emission shape the NV side already
// uses in cuda_ptx_generated.h. Until then this emitter exists purely to
// (a) set up a 'libcudacxx.test.atomics.codegen.hip.diff' regen-vs-in-tree
// drift test and (b) stop calling a hand-maintained file 'generated'.
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

  // load
  out << fmt::format(R"XXX(
template<class _Type>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type& __dst, int __memorder, {0}) {{
    if (__cuda_load_weak_if_local(__ptr, &__dst, sizeof(_Type))) return;
    __dst = __hip_atomic_load(__ptr, __memorder, {1});
}}
)XXX",
                     scope_tag, scope_macro);

  // store
  out << fmt::format(R"XXX(
template<class _Type>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type& __val, int __memorder, {0}) {{
    if (__cuda_store_weak_if_local(__ptr, &__val, sizeof(_Type))) return;
    __hip_atomic_store(__ptr, __val, __memorder, {1});
}}
)XXX",
                     scope_tag, scope_macro);

  // compare_exchange
  if (cas_branches_on_weak)
  {
    out << fmt::format(R"XXX(
template<class _Type>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type __desired, bool __is_weak, int __success_memorder, int __failure_memorder, {0}) {{
    bool __success;
    if (__cuda_compare_exchange_weak_if_local(__ptr, __expected, &__desired, &__success)) return __success;
    if(__is_weak)
        return __hip_atomic_compare_exchange_weak(__ptr, __expected, __desired, __success_memorder, __failure_memorder, {1});
    else
        return __hip_atomic_compare_exchange_strong(__ptr, __expected, __desired, __success_memorder, __failure_memorder, {1});
}}
)XXX",
                       scope_tag, scope_macro);
  }
  else
  {
    out << fmt::format(R"XXX(
template<class _Type>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type __desired, bool, int __success_memorder, int __failure_memorder, {0}) {{
    bool __success;
    if (__cuda_compare_exchange_weak_if_local(__ptr, __expected, &__desired, &__success)) return __success;
    return __hip_atomic_compare_exchange_weak(__ptr, __expected, __desired, __success_memorder, __failure_memorder, {1});
}}
)XXX",
                       scope_tag, scope_macro);
  }

  // exchange
  out << fmt::format(R"XXX(
template<class _Type>
__device__ void __atomic_exchange_cuda(volatile _Type* __ptr, _Type& __old, _Type __new, int __memorder, {0}) {{
    if (__cuda_exchange_weak_if_local(__ptr, &__new, &__old)) return;
    __old = __hip_atomic_exchange(__ptr, __new, __memorder, {1});
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

inline void FormatHip(std::ostream& out)
{
  FormatHipHeader(out);
  FormatHipScope(out,
                 /*scope_tag=*/"__thread_scope_block_tag",
                 /*fence_func=*/"__threadfence_block",
                 /*scope_macro=*/"__HIP_MEMORY_SCOPE_WORKGROUP",
                 /*has_ptr_fetch_sub=*/false,
                 /*cas_branches_on_weak=*/false);
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
