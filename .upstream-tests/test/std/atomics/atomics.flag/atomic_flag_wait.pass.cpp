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
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template<template<typename, typename> class Selector>
__host__ __device__
void test()
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __shared__
#endif
    hip::std::atomic_flag * t;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (threadIdx.x == 0) {
#endif
    t = new hip::std::atomic_flag();
    hip::std::atomic_flag_clear(t);
    hip::std::atomic_flag_wait(t, true);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    }
    __syncthreads();
#endif

    auto agent_notify = LAMBDA (){
        assert(hip::std::atomic_flag_test_and_set(t) == false);
        hip::std::atomic_flag_notify_one(t);
    };

    auto agent_wait = LAMBDA (){
        hip::std::atomic_flag_wait(t, false);
    };

    concurrent_agents_launch(agent_notify, agent_wait);

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __shared__
#endif
    volatile hip::std::atomic_flag * vt;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (threadIdx.x == 0) {
#endif
    vt = new hip::std::atomic_flag();
    hip::std::atomic_flag_clear(vt);
    hip::std::atomic_flag_wait(vt, true);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    }
    __syncthreads();
#endif

    auto agent_notify_v = LAMBDA (){
        assert(hip::std::atomic_flag_test_and_set(vt) == false);
        hip::std::atomic_flag_notify_one(vt);
    };

    auto agent_wait_v = LAMBDA (){
        hip::std::atomic_flag_wait(vt, false);
    };

    concurrent_agents_launch(agent_notify_v, agent_wait_v);
}

int main(int, char**)
{
#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    gpu_thread_count = 2;
#endif

    test<shared_memory_selector>();

  return 0;
}
