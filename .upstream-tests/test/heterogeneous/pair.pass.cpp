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

// UNSUPPORTED: nvrtc

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include "helpers.h"

#include <hip/std/utility>
#include <hip/std/cassert>

struct pod {
    char val[10];
};

using pair_t = hip::std::pair<int, pod>;

template<int N>
struct Write
{
    using async = hip::std::false_type;

    template <typename Pair>
    __host__ __device__
    static void perform(Pair &p)
    {
        hip::std::get<0>(p) = N;
        hip::std::get<1>(p).val[0] = N;
    }
};

template<int N>
struct Read
{
    using async = hip::std::false_type;

    template <typename Pair>
    __host__ __device__
    static void perform(Pair &p)
    {
        assert(hip::std::get<0>(p) == N);
        assert(hip::std::get<1>(p).val[0] == N);
    }
};

using w_r_w_r = performer_list<
  Write<10>,
  Read<10>,
  Write<30>,
  Read<30>
>;

void kernel_invoker()
{
    pair_t p(0, {0});
    validate_not_movable<
        pair_t,
        w_r_w_r
    >(p);
}

int main(int arg, char ** argv)
{
#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    kernel_invoker();
#endif

    return 0;
}
