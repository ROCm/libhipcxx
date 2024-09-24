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
// UNSUPPORTED: windows && pre-sm-70, nvrtc

// <cuda/std/atomic>

#include <hip/std/atomic>
#include <hip/std/type_traits>
#include <hip/std/cassert>
#ifdef __HIP__
#include <hip/hip_fp16.h>
#endif

#include "test_macros.h"
#include "atomic_helpers.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename Selector, hip::thread_scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    {
        typedef hip::atomic<T> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        t.fetch_min(4);
    }
    {
        typedef hip::atomic<T> A;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & t = *sel.construct();
        t.fetch_max(4);
    }
    T tmp = T(0);
    {
        hip::atomic_ref<T> t(tmp);
        t.fetch_min(4);
    }
    {
        hip::atomic_ref<T> t(tmp);
        t.fetch_max(4);
    }
  }
};

int main(int, char**)
{
#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) || __CUDA_ARCH__ >= 700 
    TestFn<__half, local_memory_selector, hip::thread_scope::thread_scope_thread>()();
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    TestFn<__half, shared_memory_selector, hip::thread_scope::thread_scope_thread>()();
    TestFn<__half, global_memory_selector, hip::thread_scope::thread_scope_thread>()();
#endif

  return 0;
}
