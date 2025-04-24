//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

#include <cuda/std/semaphore>

template<int N>
struct release
{
    using async = cuda::std::true_type;

    template<typename Semaphore>
    __host__ __device__
    static void perform(Semaphore & semaphore)
    {
        semaphore.release(N);
    }
};

struct acquire
{
    using async = cuda::std::true_type;

    template<typename Semaphore>
    __host__ __device__
    static void perform(Semaphore & semaphore)
    {
        semaphore.acquire();
    }
};

using a_a_r2 = performer_list<
    acquire,
    acquire,
    release<2>
>;

using a_a_a_r1_r2 = performer_list<
    acquire,
    acquire,
    acquire,
    release<1>,
    release<2>
>;

using a_r3_a_a = performer_list<
    acquire,
    release<3>,
    acquire,
    acquire
>;

void kernel_invoker()
{
    validate_not_movable<
        cuda::std::counting_semaphore<3>,
        a_a_r2
    >(0);
    validate_not_movable<
        cuda::counting_semaphore<cuda::thread_scope_system, 3>,
        a_a_r2
    >(0);

    validate_not_movable<
        cuda::std::counting_semaphore<3>,
        a_a_a_r1_r2
    >(0);
    validate_not_movable<
        cuda::counting_semaphore<cuda::thread_scope_system, 3>,
        a_a_a_r1_r2
    >(0);

    validate_not_movable<
        cuda::std::counting_semaphore<3>,
        a_r3_a_a
    >(0);
    validate_not_movable<
        cuda::counting_semaphore<cuda::thread_scope_system, 3>,
        a_r3_a_a
    >(0);

}

int main(int arg, char ** argv)
{
    NV_IF_TARGET(NV_IS_HOST,(
        kernel_invoker();
    ))

    return 0;
}
