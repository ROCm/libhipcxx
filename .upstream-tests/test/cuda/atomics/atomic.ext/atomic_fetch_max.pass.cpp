//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
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
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

#include <hip/std/atomic>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename Selector, hip::thread_scope ThreadScope, bool Signed = hip::std::is_signed<T>::value>
struct TestFn {
  __host__ __device__
  void operator()() const {
    // Test greater
    {
        typedef hip::atomic<T> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        t = T(1);
        assert(t.fetch_max(2) == T(1));
        assert(t.load() == T(2));
    }
    {
        typedef hip::atomic<T> A;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & t = *sel.construct();
        t = T(1);
        assert(t.fetch_max(2) == T(1));
        assert(t.load() == T(2));
    }
    // Test not greater
    {
        typedef hip::atomic<T> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        t = T(3);
        assert(t.fetch_max(2) == T(3));
        assert(t.load() == T(3));
    }
    {
        typedef hip::atomic<T> A;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & t = *sel.construct();
        t = T(3);
        assert(t.fetch_max(2) == T(3));
        assert(t.load() == T(3));
    }
  }
};

template <class T, template<typename, typename> typename Selector, hip::thread_scope ThreadScope>
struct TestFn<T, Selector, ThreadScope, true> {
  __host__ __device__
  void operator()() const {
    // Call unsigned tests
    TestFn<T, Selector, ThreadScope, false>()();
    // Test greater, but with signed math
    {
        typedef hip::atomic<T> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        t = T(-5);
        assert(t.fetch_max(-1) == T(-5));
        assert(t.load() == T(-1));
    }
    {
        typedef hip::atomic<T> A;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & t = *sel.construct();
        t = T(-5);
        assert(t.fetch_max(-1) == T(-5));
        assert(t.load() == T(-1));
    }
    // Test not greater
    {
        typedef hip::atomic<T> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        t = T(-1);
        assert(t.fetch_max(-5) == T(-1));
        assert(t.load() == T(-1));
    }
    {
        typedef hip::atomic<T> A;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & t = *sel.construct();
        t = T(-1);
        assert(t.fetch_max(-5) == T(-1));
        assert(t.load() == T(-1));
    }
  }
};

template <class T, template <typename, typename> typename Selector, hip::thread_scope ThreadScope>
struct TestFnDispatch {
  __host__ __device__
  void operator()() const {
    TestFn<T, Selector, ThreadScope>()();
  }
};

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    TestEachIntegralType<TestFnDispatch, local_memory_selector>()();
    TestEachFloatingPointType<TestFnDispatch, local_memory_selector>()();
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    TestEachIntegralType<TestFnDispatch, shared_memory_selector>()();
    TestEachFloatingPointType<TestFnDispatch, shared_memory_selector>()();
    TestEachIntegralType<TestFnDispatch, global_memory_selector>()();
    TestEachFloatingPointType<TestFnDispatch, global_memory_selector>()();
#endif

  return 0;
}
