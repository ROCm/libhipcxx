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

// template <class T>
//     bool
//     atomic_is_lock_free(const volatile atomic<T>* obj);
//
// template <class T>
//     bool
//     atomic_is_lock_free(const atomic<T>* obj);

#include <hip/std/atomic>
#include <hip/std/cassert>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename, hip::thread_scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    typedef hip::std::atomic<T> A;
    A t{};
    bool b1 = hip::std::atomic_is_lock_free(static_cast<const A*>(&t));
    volatile A vt{};
    bool b2 = hip::std::atomic_is_lock_free(static_cast<const volatile A*>(&vt));
    assert(b1 == b2);
  }
};

struct A
{
    char _[4];
};

int main(int, char**)
{
    TestFn<A, local_memory_selector, hip::thread_scope_system>()();
    TestEachAtomicType<TestFn, local_memory_selector>()();

  return 0;
}
