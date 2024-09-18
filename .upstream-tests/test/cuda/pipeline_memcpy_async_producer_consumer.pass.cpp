//===----------------------------------------------------------------------===//
//
// Part of libhip++ (derived from libcu++),
// the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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
// UNSUPPORTED: hipcc

// UNSUPPORTED: pre-sm-70

#include <cooperative_groups.h>
#include <hip/pipeline>

#include "cuda_space_selector.h"

template <class T, hip::thread_scope PipelineScope>
__device__ __noinline__
void test_producer(T * dest, T * source, hip::pipeline<PipelineScope> & pipe)
{
    pipe.producer_acquire();
    hip::memcpy_async(static_cast<void*>(&dest[0]), static_cast<void*>(&source[0]), sizeof(T), pipe);
    pipe.producer_commit();

    pipe.producer_acquire();
    hip::memcpy_async(static_cast<void*>(&dest[1]), static_cast<void*>(&source[1]), sizeof(T), pipe);
    pipe.producer_commit();

    pipe.producer_acquire();
    hip::memcpy_async(static_cast<void*>(&dest[2]), static_cast<void*>(&source[2]), sizeof(T), pipe);
    pipe.producer_commit();
}

template <class T, hip::thread_scope PipelineScope>
__device__ __noinline__
void test_consumer(T * dest, T * source, hip::pipeline<PipelineScope> & pipe)
{
    pipe.consumer_wait();
    assert(source[0] == 12);
    assert(dest[0] == 12);
    pipe.consumer_release();

    pipe.consumer_wait();
    assert(source[1] == 24);
    assert(dest[1] == 24);
    pipe.consumer_release();

    pipe.consumer_wait();
    assert(source[2] == 36);
    assert(dest[2] == 36);
    pipe.consumer_release();
}

template <class T,
    template<typename, typename> class PipelineSelector,
    hip::thread_scope PipelineScope,
    uint8_t PipelineStages
>
__device__ __noinline__
void test_fully_specialized()
{
    __shared__ T dest[3];
    __shared__ T * source;

    PipelineSelector<hip::pipeline_shared_state<PipelineScope, PipelineStages>, constructor_initializer> pipe_state_sel;
    hip::pipeline_shared_state<PipelineScope, PipelineStages> * pipe_state = pipe_state_sel.construct();

    auto group = cooperative_groups::this_thread_block();

    if (group.thread_rank() == 0) {
        source = new T[3];
        for (size_t i = 0; i < 3; ++i) {
            source[i] = 12 * (i + 1);
        }
    }
    group.sync();

    {
        if (group.thread_rank() == 0) {
            memset(dest, 0, sizeof(T) * 3);
        }
        group.sync();

        const hip::pipeline_role role = (group.thread_rank() % 2) ? hip::pipeline_role::producer
                                                                   : hip::pipeline_role::consumer;
        hip::pipeline<PipelineScope> pipe = make_pipeline(group, pipe_state, role);

        if (role == hip::pipeline_role::producer) {
            test_producer(dest, source, pipe);
       } else {
           test_consumer(dest, source, pipe);
       }

        pipe.quit();
    }

    group.sync();

    {
        if (group.thread_rank() == 0) {
            memset(dest, 0, sizeof(T) * 3);
        }
        group.sync();

        const hip::pipeline_role role = (group.thread_rank() < 32) ? hip::pipeline_role::producer
                                                                    : hip::pipeline_role::consumer;
        hip::pipeline<PipelineScope> pipe = make_pipeline(group, pipe_state, group.size() / 2);

        if (role == hip::pipeline_role::producer) {
            test_producer(dest, source, pipe);
        } else {
            test_consumer(dest, source, pipe);
        }

        pipe.quit();
    }
}

template <class T,
    template<typename, typename> class PipelineSelector,
    hip::thread_scope PipelineScope
>
__host__ __device__ __noinline__
void test_select_stages()
{
    test_fully_specialized<T, PipelineSelector, PipelineScope, 1>();
    test_fully_specialized<T, PipelineSelector, PipelineScope, 8>();
}

template <class T,
    template<typename, typename> class PipelineSelector
>
__host__ __device__ __noinline__
void test_select_scope()
{
    test_select_stages<T, PipelineSelector, hip::thread_scope_block>();
    test_select_stages<T, PipelineSelector, hip::thread_scope_device>();
    test_select_stages<T, PipelineSelector, hip::thread_scope_system>();
}

template <class T>
__host__ __device__ __noinline__
void test_select_pipeline()
{
    test_select_scope<T, shared_memory_selector>();
    test_select_scope<T, global_memory_selector>();
}

int main(int argc, char ** argv)
{
#ifndef __CUDA_ARCH__
    gpu_thread_count = 64;
#else
    test_select_pipeline<uint8_t>();
    test_select_pipeline<uint16_t>();
    test_select_pipeline<uint32_t>();
    test_select_pipeline<uint64_t>();
#endif

    return 0;
}
