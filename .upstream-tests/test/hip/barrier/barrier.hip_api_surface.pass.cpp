//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// UNSUPPORTED: nvcc, nvhpc, nvc++

// <cuda/barrier>
// Verify the class-definition conformance of hip::barrier on AMD targets for
// all supported scopes. Every check operates on the class declaration

#include <cuda/barrier>
#include <cuda/std/chrono>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cstdlib>
#include "hip_barrier_test_utils.h"

using namespace hip_test;

// is_detected<Op, T>: true_type iff Op<T> is well-formed.
// Each trait below is just a type alias naming the expression to probe.
template <class...>
using void_t = void;

template <template <class> class Op, class T, class = void>
struct is_detected : hip::std::false_type
{};

template <template <class> class Op, class T>
struct is_detected<Op, T, void_t<Op<T>>> : hip::std::true_type
{};

template <class B>
using wait_parity_expr = decltype(hip::std::declval<const B&>().wait_parity(false));

template <class B>
using try_wait_for_expr = decltype(hip::std::declval<const B&>().try_wait_for(
  hip::std::declval<typename B::arrival_token>(),
  hip::std::declval<hip::std::chrono::nanoseconds>()));

template <class B>
using try_wait_until_expr = decltype(hip::std::declval<const B&>().try_wait_until(
  hip::std::declval<typename B::arrival_token>(),
  hip::std::declval<hip::std::chrono::high_resolution_clock::time_point>()));

template <class B>
using try_wait_parity_for_expr = decltype(hip::std::declval<const B&>().try_wait_parity_for(
  false, hip::std::declval<hip::std::chrono::nanoseconds>()));

template <class B>
using try_wait_parity_until_expr = decltype(hip::std::declval<const B&>().try_wait_parity_until(
  false, hip::std::declval<hip::std::chrono::high_resolution_clock::time_point>()));

/// Per-type surface check
/// @tparam Scope scope of barrier type under test
/// @tparam HasBlockScopeExtensions
///    controls whether the five methods that are exclusive to 
///    barrier<thread_scope_block> should be present (true) or absent (false).
/// @tparam CompFuncT type of completion function of the barrier type under test
template <hip::thread_scope Scope, bool HasBlockScopeExtensions, typename CompFuncT = hip::std::__empty_completion>
struct api_surface_checks
{
  using B = hip::barrier<Scope, CompFuncT>;
  static bool constexpr isEmptyCompletion = hip::std::is_same_v<hip::std::__empty_completion, CompFuncT>;
  using ExpectedArrivalTokenType =
    hip::std::conditional_t<isEmptyCompletion, hip::std::uint64_t, bool>;
  static_assert(hip::std::is_same_v<typename B::arrival_token, ExpectedArrivalTokenType>,
                "arrival_token type mismatch");

  static_assert(hip::std::is_default_constructible<B>::value, "");
  // Bs are non-copyable; a copy would alias the same atomic state word,
  // producing data races on arrive() and wait().
  static_assert(!hip::std::is_copy_constructible_v<B>,
                "barrier must not be copy-constructible");
  static_assert(!hip::std::is_copy_assignable_v<B>,
                "barrier must not be copy-assignable");

  // HIP does not allow launching thread blocks larger than 1024
  // At block scope, this is a sensible minimum for max().
  // Larger scopes should not have a smaller max() than block scope.
  static constexpr int kMaxBlockSize = 1024;
  static_assert(B::max() >= kMaxBlockSize);

  // wait_parity is available on all scopes without completion functions.
  static_assert(is_detected<wait_parity_expr, B>::value == (isEmptyCompletion && Scope != hip::thread_scope_thread),
                "wait_parity must be present for empty completion function barriers at all scopes but thread_scope_thread");

  // The four timed-wait overloads are block-scope-only.
  static_assert(is_detected<try_wait_for_expr, B>::value == HasBlockScopeExtensions,
                "try_wait_for availability mismatch");
  static_assert(is_detected<try_wait_until_expr, B>::value == HasBlockScopeExtensions,
                "try_wait_until availability mismatch");
  static_assert(is_detected<try_wait_parity_for_expr, B>::value == HasBlockScopeExtensions,
                "try_wait_parity_for availability mismatch");
  static_assert(is_detected<try_wait_parity_until_expr, B>::value == HasBlockScopeExtensions,
                "try_wait_parity_until availability mismatch");
};

using CompFuncT = decltype(&barrier_no_op_completion);
template struct api_surface_checks<hip::thread_scope_thread, false>;
template struct api_surface_checks<hip::thread_scope_thread, false, CompFuncT>;
template struct api_surface_checks<hip::thread_scope_block, true>;
template struct api_surface_checks<hip::thread_scope_block, false, CompFuncT>;
template struct api_surface_checks<hip::thread_scope_device, false>;
template struct api_surface_checks<hip::thread_scope_device, false, CompFuncT>;
template struct api_surface_checks<hip::thread_scope_system, false>;
template struct api_surface_checks<hip::thread_scope_system, false, CompFuncT>;

__device__ int test() {
// If barrier utilizes AMD LDS HW barriers max() is a HW spec
#if defined(__HIP_PLATFORM_AMD__) && defined(__gfx1250__)
  __shared__ hip::barrier<hip::thread_scope_block> b;  
  return b.max() == 0xFFFF ? EXIT_SUCCESS : EXIT_FAILURE;
#endif
  return EXIT_SUCCESS;
}

int main(int, char**)
{
  int result = EXIT_SUCCESS;
  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
