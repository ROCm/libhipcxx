//===----------------------------------------------------------------------===//
//
// Part of the libhip++ Project (derived from libcu++),
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
// UNSUPPORTED: nvrtc, pre-sm-70

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include "helpers.h"

#include <hip/barrier>

__managed__ bool completed_from_host = false;
__managed__ bool completed_from_device = false;

template<typename Barrier>
struct barrier_and_token
{
    using barrier_t = Barrier;
    using token_t = typename barrier_t::arrival_token;

    barrier_t barrier;
    hip::std::atomic<token_t> token{token_t{}};
    hip::std::atomic<bool> token_set{false};

    template<typename ...Args>
    __host__ __device__
    barrier_and_token(Args && ...args) : barrier{ hip::std::forward<Args>(args)... }
    {
    }
};

template<template<typename> typename Barrier>
struct barrier_and_token_with_completion
{
    struct completion_t
    {
        hip::std::atomic<bool> & completed;

        __host__ __device__
        void operator()() const
        {
            assert(completed.load() == false);
            completed.store(true);

#ifdef __CUDA_ARCH__
            completed_from_device = true;
#else
            completed_from_host = true;
#endif
        }
    };

    using barrier_t = Barrier<completion_t>;
    using token_t = typename barrier_t::arrival_token;

    barrier_t barrier;
    hip::std::atomic<token_t> token{token_t{}};
    hip::std::atomic<bool> token_set{false};
    hip::std::atomic<bool> completed{false};

    template<typename Arg>
    __host__ __device__
    barrier_and_token_with_completion(Arg && arg)
        : barrier{ std::forward<Arg>(arg), completion_t{ completed } }
    {
    }
};

struct barrier_arrive
{
    using async = hip::std::true_type;

    template<typename Data>
    __host__ __device__
    static void perform(Data & data)
    {
        data.token.store(data.barrier.arrive(), hip::std::memory_order_release);
        data.token_set.store(true, hip::std::memory_order_release);
        data.token_set.notify_all();
    }
};

struct barrier_wait
{
    using async = hip::std::true_type;

    template<typename Data>
    __host__ __device__
    static void perform(Data & data)
    {
        while (data.token_set.load(hip::std::memory_order_acquire) == false)
        {
            data.token_set.wait(false);
        }
        data.barrier.wait(data.token);
    }
};

struct barrier_arrive_and_wait
{
    using async = hip::std::true_type;

    template<typename Data>
    __host__ __device__
    static void perform(Data & data)
    {
        data.barrier.arrive_and_wait();
    }
};

struct validate_completion_result
{
    template<typename Data>
    __host__ __device__
    static void perform(Data & data)
    {
        assert(data.completed.load(hip::std::memory_order_acquire) == true);
        data.completed.store(false, hip::std::memory_order_release);
    }
};

struct clear_token
{
    template<typename Data>
    __host__ __device__
    static void perform(Data & data)
    {
        data.token_set.store(false, hip::std::memory_order_release);
    }
};

using a_aw_w = performer_list<
    barrier_arrive,
    barrier_arrive_and_wait,
    barrier_wait,
    async_tester_fence,
    clear_token
>;

using aw_aw = performer_list<
    barrier_arrive_and_wait,
    barrier_arrive_and_wait
>;

using a_w_aw = performer_list<
    barrier_arrive,
    barrier_wait,
    barrier_arrive_and_wait,
    async_tester_fence,
    clear_token
>;

using a_w_a_w = performer_list<
    barrier_arrive,
    barrier_wait,
    barrier_arrive,
    barrier_wait,
    async_tester_fence,
    clear_token
>;

using completion_performers_a = performer_list<
    clear_token,
    barrier_arrive,
    barrier_arrive_and_wait,
    async_tester_fence,
    validate_completion_result,
    barrier_wait
>;

using completion_performers_b = performer_list<
    clear_token,
    barrier_arrive,
    barrier_arrive_and_wait,
    async_tester_fence,
    validate_completion_result
>;

template<typename Completion>
using cuda_barrier_system = hip::barrier<hip::thread_scope_system, Completion>;

void kernel_invoker()
{
    validate_not_movable<
        barrier_and_token<hip::std::barrier<>>,
        a_aw_w>(2);
    validate_not_movable<
        barrier_and_token<hip::barrier<hip::thread_scope_system>>,
        a_aw_w>(2);

    validate_not_movable<
        barrier_and_token<hip::std::barrier<>>,
        aw_aw>(2);
    validate_not_movable<
        barrier_and_token<hip::barrier<hip::thread_scope_system>>,
        aw_aw>(2);

    validate_not_movable<
        barrier_and_token<hip::std::barrier<>>,
        a_w_aw>(2);
    validate_not_movable<
        barrier_and_token<hip::barrier<hip::thread_scope_system>>,
        a_w_aw>(2);

    validate_not_movable<
        barrier_and_token<hip::std::barrier<>>,
        a_w_a_w>(2);
    validate_not_movable<
        barrier_and_token<hip::barrier<hip::thread_scope_system>>,
        a_w_a_w>(2);

    validate_not_movable<
        barrier_and_token_with_completion<hip::std::barrier>,
        completion_performers_a>(2);
    validate_not_movable<
        barrier_and_token_with_completion<cuda_barrier_system>,
        completion_performers_a>(2);

    validate_not_movable<
        barrier_and_token_with_completion<hip::std::barrier>,
        completion_performers_b>(2);
    validate_not_movable<
        barrier_and_token_with_completion<cuda_barrier_system>,
        completion_performers_b>(2);
}

int main(int arg, char ** argv)
{
#ifndef __CUDA_ARCH__
    kernel_invoker();

    if (check_managed_memory_support(true))
    {
        assert(completed_from_host);
    }
    assert(completed_from_device);
#endif

    return 0;
}

