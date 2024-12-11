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
//  ... test crashes clang

// <cuda/std/atomic>

// template <class Integral>
//     Integral
//     atomic_fetch_add(volatile atomic<Integral>* obj, Integral op);
//
// template <class Integral>
//     Integral
//     atomic_fetch_add(atomic<Integral>* obj, Integral op);
//
// template <class T>
//     T*
//     atomic_fetch_add(volatile atomic<T*>* obj, ptrdiff_t op);
//
// template <class T>
//     T*
//     atomic_fetch_add(atomic<T*>* obj, ptrdiff_t op);

#include <hip/std/atomic>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename Selector, hip::thread_scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    {
        typedef hip::std::atomic<T> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        hip::std::atomic_init(&t, T(1));
        assert(hip::std::atomic_fetch_add(&t, T(2)) == T(1));
        assert(t == T(3));
    }
    {
        typedef hip::std::atomic<T> A;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & t = *sel.construct();
        hip::std::atomic_init(&t, T(1));
        assert(hip::std::atomic_fetch_add(&t, T(2)) == T(1));
        assert(t == T(3));
    }
  }
};

template <class T, template<typename, typename> typename Selector>
__host__ __device__
void testp()
{
    {
        typedef hip::std::atomic<T> A;
        typedef typename hip::std::remove_pointer<T>::type X;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        hip::std::atomic_init(&t, T(1*sizeof(X)));
        assert(hip::std::atomic_fetch_add(&t, 2) == T(1*sizeof(X)));
        assert(t == T(3*sizeof(X)));
    }
    {
        typedef hip::std::atomic<T> A;
        typedef typename hip::std::remove_pointer<T>::type X;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & t = *sel.construct();
        hip::std::atomic_init(&t, T(1*sizeof(X)));
        assert(hip::std::atomic_fetch_add(&t, 2) == T(1*sizeof(X)));
        assert(t == T(3*sizeof(X)));
    }
}

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    TestEachIntegralType<TestFn, local_memory_selector>()();
    TestEachFloatingPointType<TestFn, local_memory_selector>()();
    testp<int*, local_memory_selector>();
    testp<const int*, local_memory_selector>();
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    TestEachIntegralType<TestFn, shared_memory_selector>()();
    TestEachFloatingPointType<TestFn, shared_memory_selector>()();
    testp<int*, shared_memory_selector>();
    testp<const int*, shared_memory_selector>();
    TestEachIntegralType<TestFn, global_memory_selector>()();
    TestEachFloatingPointType<TestFn, global_memory_selector>()();
    testp<int*, global_memory_selector>();
    testp<const int*, global_memory_selector>();
#endif

  return 0;
}
