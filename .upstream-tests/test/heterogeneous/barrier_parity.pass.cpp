//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
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

#include <atomic>
#include <hip/barrier>

template<typename Barrier>
struct barrier_and_token
{
    using barrier_t = Barrier;
    using token_t = typename barrier_t::arrival_token;

    barrier_t barrier;
    hip::std::atomic<bool> parity_waiting{false};

    template<typename ...Args>
    __host__ __device__
    barrier_and_token(Args && ...args) : barrier{ hip::std::forward<Args>(args)... }
    {
    }
};

struct barrier_arrive_and_wait
{
    using async = hip::std::true_type;

    template<typename Data>
    __host__ __device__
    static void perform(Data & data)
    {
        while (data.parity_waiting.load(hip::std::memory_order_acquire) == false)
        {
            data.parity_waiting.wait(false);
        }
        data.barrier.arrive_and_wait();
    }
};

template <bool Phase>
struct barrier_parity_wait
{
    using async = hip::std::true_type;

    template<typename Data>
    __host__ __device__
    static void perform(Data & data)
    {
        data.parity_waiting.store(true, hip::std::memory_order_release);
        data.parity_waiting.notify_all();
        data.barrier.wait_parity(Phase);
    }
};

struct clear_token
{
    template<typename Data>
    __host__ __device__
    static void perform(Data & data)
    {
        data.parity_waiting.store(false, hip::std::memory_order_release);
    }
};

using aw_aw_pw = performer_list<
    barrier_parity_wait<false>,
    barrier_arrive_and_wait,
    barrier_arrive_and_wait,
    async_tester_fence,
    clear_token,
    barrier_parity_wait<true>,
    barrier_arrive_and_wait,
    barrier_arrive_and_wait,
    async_tester_fence,
    clear_token
>;

void kernel_invoker()
{
    validate_not_movable<
        barrier_and_token<hip::std::barrier<>>,
        aw_aw_pw
      >(2);
    validate_not_movable<
        barrier_and_token<hip::barrier<hip::thread_scope_system>>,
        aw_aw_pw
      >(2);
}

int main(int arg, char ** argv)
{
#ifndef __CUDA_ARCH__
    kernel_invoker();
#endif

    return 0;
}

