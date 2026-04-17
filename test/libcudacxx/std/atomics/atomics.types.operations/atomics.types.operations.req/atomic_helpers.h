//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#ifndef ATOMIC_HELPERS_H
#define ATOMIC_HELPERS_H

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "cuda_space_selector.h"
#include "test_macros.h"

struct UserAtomicType
{
  int i;

  TEST_FUNC explicit UserAtomicType(int d = 0) noexcept
      : i(d)
  {}

  TEST_FUNC friend bool operator==(const UserAtomicType& x, const UserAtomicType& y)
  {
    return x.i == y.i;
  }
};

// NOTE(HIP/AMD): the upstream gate
//   _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
// defaults the `Scope` template argument to `thread_scope_system`
// only on platforms that support system-scope atomics: any host
// pass, and NVPTX sm_60+ (which is when CUDA grew system-scope
// atomic instructions). On HIP-device every supported AMDGCN
// target (gfx90a/94x/10/11/12/950) has native system-scope atomics,
// so add `_CCCL_HIP_COMPILATION()` to the OR-chain to give HIP-device
// passes the same default. Without it, HIP-device callers
// instantiating `TestEachIntegralType<...>` would have to spell out
// the third template argument explicitly (it has no default), which
// every existing call site does NOT do -- removing the HIP arm
// here would make the upstream call sites fail to compile under HIP.
template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          >
struct TestEachIntegralType
{
  TEST_FUNC void operator()() const
  {
    TestFunctor<char, Selector, Scope>()();
    TestFunctor<signed char, Selector, Scope>()();
    TestFunctor<unsigned char, Selector, Scope>()();
    TestFunctor<short, Selector, Scope>()();
    TestFunctor<unsigned short, Selector, Scope>()();
    TestFunctor<int, Selector, Scope>()();
    TestFunctor<unsigned int, Selector, Scope>()();
    TestFunctor<long, Selector, Scope>()();
    TestFunctor<unsigned long, Selector, Scope>()();
    TestFunctor<long long, Selector, Scope>()();
    TestFunctor<unsigned long long, Selector, Scope>()();
    TestFunctor<wchar_t, Selector, Scope>();
    TestFunctor<char16_t, Selector, Scope>()();
    TestFunctor<char32_t, Selector, Scope>()();
    TestFunctor<int8_t, Selector, Scope>()();
    TestFunctor<uint8_t, Selector, Scope>()();
    TestFunctor<int16_t, Selector, Scope>()();
    TestFunctor<uint16_t, Selector, Scope>()();
    TestFunctor<int32_t, Selector, Scope>()();
    TestFunctor<uint32_t, Selector, Scope>()();
    TestFunctor<int64_t, Selector, Scope>()();
    TestFunctor<uint64_t, Selector, Scope>()();
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          >
struct TestEachFloatingPointType
{
  TEST_FUNC void operator()() const
  {
    TestFunctor<float, Selector, Scope>()();
    TestFunctor<double, Selector, Scope>()();
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          >
struct TestEachAtomicType
{
  TEST_FUNC void operator()() const
  {
    TestEachIntegralType<TestFunctor, Selector, Scope>()();
    TestEachFloatingPointType<TestFunctor, Selector, Scope>()();
    TestFunctor<UserAtomicType, Selector, Scope>()();
    TestFunctor<int*, Selector, Scope>()();
    TestFunctor<const int*, Selector, Scope>()();
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          >
struct TestEachIntegralRefType
{
  TEST_FUNC void operator()() const
  {
    TestFunctor<int, Selector, Scope>()();
    TestFunctor<unsigned int, Selector, Scope>()();
    TestFunctor<long, Selector, Scope>()();
    TestFunctor<unsigned long, Selector, Scope>()();
    TestFunctor<long long, Selector, Scope>()();
    TestFunctor<unsigned long long, Selector, Scope>()();
    TestFunctor<char32_t, Selector, Scope>()();
    TestFunctor<int32_t, Selector, Scope>()();
    TestFunctor<uint32_t, Selector, Scope>()();
    TestFunctor<int64_t, Selector, Scope>()();
    TestFunctor<uint64_t, Selector, Scope>()();
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          >
struct TestEachFLoatingPointRefType
{
  TEST_FUNC void operator()() const
  {
    TestFunctor<float, Selector, Scope>()();
    TestFunctor<double, Selector, Scope>()();
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector = shared_memory_selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600 || _CCCL_HIP_COMPILATION()
          >
struct TestEachAtomicRefType
{
  TEST_FUNC void operator()() const
  {
    TestEachIntegralRefType<TestFunctor, Selector, Scope>()();
    TestEachFLoatingPointRefType<TestFunctor, Selector, Scope>()();
    TestFunctor<UserAtomicType, Selector, Scope>()();
    TestFunctor<int*, Selector, Scope>()();
    TestFunctor<const int*, Selector, Scope>()();
  }
};

#endif // ATOMIC_HELPER_H
