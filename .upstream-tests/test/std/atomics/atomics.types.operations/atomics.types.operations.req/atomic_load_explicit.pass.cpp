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
//  ... assertion fails line 31

// <cuda/std/atomic>

// template <class T>
//     T
//     atomic_load_explicit(const volatile atomic<T>* obj, memory_order m);
//
// template <class T>
//     T
//     atomic_load_explicit(const atomic<T>* obj, memory_order m);

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
    typedef hip::std::atomic<T> A;
    Selector<A, constructor_initializer> sel;
    A & t = *sel.construct();
    hip::std::atomic_init(&t, T(1));
    assert(hip::std::atomic_load_explicit(&t, hip::std::memory_order_seq_cst) == T(1));
    Selector<volatile A, constructor_initializer> vsel;
    volatile A & vt = *vsel.construct();
    hip::std::atomic_init(&vt, T(2));
    assert(hip::std::atomic_load_explicit(&vt, hip::std::memory_order_seq_cst) == T(2));
  }
};

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    TestEachAtomicType<TestFn, local_memory_selector>()();
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    TestEachAtomicType<TestFn, shared_memory_selector>()();
    TestEachAtomicType<TestFn, global_memory_selector>()();
#endif

  return 0;
}
