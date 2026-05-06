//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef __LIBCUDACXX_CCCLRT_COMMON_TESTING_H__
#define __LIBCUDACXX_CCCLRT_COMMON_TESTING_H__

#include <cuda/std/detail/__config>

#include <cuda/__driver/driver_api.h>

#include <nv/target>

#include <exception> // IWYU pragma: keep
#include <sstream>

#include <c2h/catch2_test_helper.h>

#define CUDART(call) REQUIRE((call) == cudaSuccess)

__device__ inline void ccclrt_require_impl(
  bool condition, const char* condition_text, const char* filename, unsigned int linenum, const char* funcname)
{
  if (!condition)
  {
    // TODO do warp aggregate prints for easier readability?
    printf("%s:%u: %s: block: [%d,%d,%d], thread: [%d,%d,%d] Condition `%s` failed.\n",
           filename,
           linenum,
           funcname,
           blockIdx.x,
           blockIdx.y,
           blockIdx.z,
           threadIdx.x,
           threadIdx.y,
           threadIdx.z,
           condition_text);
    // NOTE(HIP/AMD): on CUDA '__trap()' is a free function intrinsic;
    // on HIP/AMDGCN there is no '__trap' free function but
    // '__builtin_trap()' compiles down to the equivalent
    // s_trap instruction (clang-hip mode).
#if defined(__HIP_PLATFORM_AMD__)
    __builtin_trap();
#else
    __trap();
#endif // !__HIP_PLATFORM_AMD__
  }
}

// There is a problem with clang-cuda and nv/target, but we don't need the device side macros yet,
// disable them for now
//
// NOTE(HIP/AMD): clang-hip exhibits the same two-pass parsing
// behaviour as clang-cuda (the device pass parses host-only function
// bodies for type checking, then discards them) so the
// 'NV_IF_ELSE_TARGET(NV_IS_DEVICE, <__device__-only>, <host>)'
// dispatch pattern below trips a 'no matching function' error inside
// host-only functions instantiated by templated tests. Take the same
// host-only definitions as the clang-cuda path on HIP.
#if _CCCL_CUDA_COMPILER(CLANG) || _CCCL_HIP_COMPILATION()
#  define CCCLRT_REQUIRE(condition)     REQUIRE(condition)
#  define CCCLRT_CHECK(condition)       CHECK(condition)
#  define CCCLRT_FAIL(message)          FAIL(message)
#  define CCCLRT_CHECK_FALSE(condition) CCCLRT_CHECK(!(condition))

#else // _CCCL_CUDA_COMPILER(CLANG) || _CCCL_HIP_COMPILATION()
#  define CCCLRT_REQUIRE(condition)                                                                           \
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,                                                                           \
                      (ccclrt_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);), \
                      (REQUIRE(condition);))

#  define CCCLRT_CHECK(condition)                                                                             \
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,                                                                           \
                      (ccclrt_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);), \
                      (CHECK(condition);))

#  define CCCLRT_FAIL(message) /*                                                                   */ \
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, /*                                                             */  \
                      (ccclrt_require_impl(false, message, __FILE__, __LINE__, __PRETTY_FUNCTION__);), \
                      (FAIL(message);))

#  define CCCLRT_CHECK_FALSE(condition) CCCLRT_CHECK(!(condition))
#endif // _CCCL_CUDA_COMPILER(CLANG) || _CCCL_HIP_COMPILATION()

// Explicit device side require macros for clang-cuda
#define CCCLRT_REQUIRE_DEVICE(condition) \
  ccclrt_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);
#define CCCLRT_CHECK_DEVICE(condition) \
  ccclrt_require_impl(condition, #condition, __FILE__, __LINE__, __PRETTY_FUNCTION__);
#define CCCLRT_FAIL_DEVICE(message)          ccclrt_require_impl(false, message, __FILE__, __LINE__, __PRETTY_FUNCTION__);
#define CCCLRT_CHECK_FALSE_DEVICE(condition) CCCLRT_CHECK_DEVICE(!(condition))

__host__ __device__ constexpr bool operator==(const dim3& lhs, const dim3& rhs) noexcept
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

namespace Catch
{
template <>
struct StringMaker<dim3>
{
  static std::string convert(dim3 const& dims)
  {
    std::ostringstream oss;
    oss << "(" << dims.x << ", " << dims.y << ", " << dims.z << ")";
    return oss.str();
  }
};
} // namespace Catch

namespace
{
namespace test
{
// NOTE(HIP/AMD): the CUDA-driver-API context-stack model (push /
// pop / GetCurrent walking back to nullptr) does not have a 1:1
// counterpart in HIP. hipCtxGetCurrent typically reports the
// primary context as "current" once the runtime has been touched,
// and trying to pop it returns hipErrorInvalidDevice (201). The
// stack-cleanliness invariant the upstream fixture enforces is a
// CUDA-only concern, so on HIP both helpers below are no-ops and
// the fixture skips the bookkeeping. Tests still get a fresh
// scope for Catch2 reporting.
inline int count_driver_stack()
{
#if !_CCCL_HIP_COMPILATION()
  if (::cuda::__driver::__ctxGetCurrent() != nullptr)
  {
    auto ctx    = ::cuda::__driver::__ctxPop();
    auto result = 1 + count_driver_stack();
    ::cuda::__driver::__ctxPush(ctx);
    return result;
  }
#endif // !_CCCL_HIP_COMPILATION()
  return 0;
}

inline void empty_driver_stack()
{
#if !_CCCL_HIP_COMPILATION()
  while (::cuda::__driver::__ctxGetCurrent() != nullptr)
  {
    ::cuda::__driver::__ctxPop();
  }
#endif // !_CCCL_HIP_COMPILATION()
}

inline int cuda_driver_version()
{
  return ::cuda::__driver::__getVersion();
}

// Needs to be a template because we use template catch2 macro
template <typename Dummy = void>
struct ccclrt_test_fixture
{
  ccclrt_test_fixture()
  {
    empty_driver_stack();
  }
  ~ccclrt_test_fixture()
  {
    CCCLRT_CHECK(count_driver_stack() == 0);
  }
};
} // namespace test
} // namespace

// Test macro that should be used in all cccl-rt tests
// It first empties the driver stack in case some other test has left it non-empty
// and then runs the test. At the end it checks if it remained empty, which ensures
// we don't accidentally initialize device 0 through CUDART usage and makes sure
// our APIs work with empty driver stack.
#define C2H_CCCLRT_TEST(NAME, TAGS, ...) C2H_TEST_WITH_FIXTURE(::test::ccclrt_test_fixture, NAME, TAGS, __VA_ARGS__)

#define C2H_CCCLRT_TEST_LIST(NAME, TAGS, ...) \
  C2H_TEST_LIST_WITH_FIXTURE(::test::ccclrt_test_fixture, NAME, TAGS, __VA_ARGS__)

#endif // __LIBCUDACXX_CCCLRT_COMMON_TESTING_H__
