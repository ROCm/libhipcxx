//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.ation.
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


// UNSUPPORTED: nvrtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include "helpers.h"

#include <hip/std/atomic>

struct clear
{
    template<typename AF>
    __host__ __device__
    static void initialize(AF & af)
    {
        af.clear();
    }
};

struct clear_tester : clear
{
    template<typename AF>
    __host__ __device__
    static void validate(AF & af)
    {
        assert(af.test_and_set() == false);
    }
};

template<bool Previous>
struct test_and_set_tester
{
    template<typename AF>
    __host__ __device__
    static void initialize(AF & af)
    {
        assert(af.test_and_set() == Previous);
    }

    template<typename AF>
    __host__ __device__
    static void validate(AF & af)
    {
        assert(af.test_and_set() == true);
    }
};

using atomic_flag_testers = tester_list<
    clear_tester,
    clear,
    test_and_set_tester<false>,
    test_and_set_tester<true>
>;

void kernel_invoker()
{
    validate_not_movable<hip::std::atomic_flag, atomic_flag_testers>();
}

int main(int argc, char ** argv)
{
#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    kernel_invoker();
#endif

    return 0;
}
