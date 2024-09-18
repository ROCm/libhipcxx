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
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: pre-sm-70
// This test is not supported yet, because chrono::system_clock has not been implemented for HIP.
// UNSUPPORTED: hipcc

// <cuda/std/atomic>

#include <hip/std/atomic>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "../atomics.types.operations.req/atomic_helpers.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename Selector, hip::thread_scope Scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    typedef hip::std::atomic<T> A;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __shared__
#endif
    A * t;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (threadIdx.x == 0) {
#endif
    t = (A *)malloc(sizeof(A));
    hip::std::atomic_init(t, T(1));
    assert(hip::std::atomic_load(t) == T(1));
    hip::std::atomic_wait(t, T(0));
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    }
    __syncthreads();
#endif

    auto agent_notify = LAMBDA (){
      hip::std::atomic_store(t, T(3));
      hip::std::atomic_notify_one(t);
    };

    auto agent_wait = LAMBDA (){
      hip::std::atomic_wait(t, T(1));
    };

    concurrent_agents_launch(agent_notify, agent_wait);

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __shared__
#endif
    volatile A * vt;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (threadIdx.x == 0) {
#endif
    vt = (volatile A *)malloc(sizeof(A));
    hip::std::atomic_init(vt, T(2));
    assert(hip::std::atomic_load(vt) == T(2));
    hip::std::atomic_wait(vt, T(1));
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    }
    __syncthreads();
#endif

    auto agent_notify_v = LAMBDA (){
      hip::std::atomic_store(vt, T(4));
      hip::std::atomic_notify_one(vt);
    };
    auto agent_wait_v = LAMBDA (){
      hip::std::atomic_wait(vt, T(2));
    };
  
    concurrent_agents_launch(agent_notify_v, agent_wait_v);
  }
};

int main(int, char**)
{
#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    gpu_thread_count = 2;
#endif

    TestEachAtomicType<TestFn, shared_memory_selector>()();

  return 0;
}
