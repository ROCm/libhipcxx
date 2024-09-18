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

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// This test verifies behavior specified by [atomics.types.operations.req]/21:
//
//     When only one memory_order argument is supplied, the value of success is
//     order, and the value of failure is order except that a value of
//     memory_order_acq_rel shall be replaced by the value memory_order_acquire
//     and a value of memory_order_release shall be replaced by the value
//     memory_order_relaxed.
//
// Clang's atomic intrinsics do this for us, but GCC's do not. We don't actually
// have visibility to see what these memory orders are lowered to, but we can at
// least check that they are lowered at all (otherwise there is a compile
// failure with GCC).

#include <hip/std/atomic>

#include "test_macros.h"
#include "cuda_space_selector.h"

template<template<typename, typename> class Selector>
__host__ __device__
void test()
{
    Selector<hip::std::atomic<int>, default_initializer> sel;
    Selector<volatile hip::std::atomic<int>, default_initializer> vsel;

    hip::std::atomic<int> & i = *sel.construct();
    volatile hip::std::atomic<int> & v = *vsel.construct();
    int exp = 0;

    (void) i.compare_exchange_weak(exp, 0, hip::std::memory_order_acq_rel);
    (void) i.compare_exchange_weak(exp, 0, hip::std::memory_order_release);
    i.compare_exchange_strong(exp, 0, hip::std::memory_order_acq_rel);
    i.compare_exchange_strong(exp, 0, hip::std::memory_order_release);

    (void) v.compare_exchange_weak(exp, 0, hip::std::memory_order_acq_rel);
    (void) v.compare_exchange_weak(exp, 0, hip::std::memory_order_release);
    v.compare_exchange_strong(exp, 0, hip::std::memory_order_acq_rel);
    v.compare_exchange_strong(exp, 0, hip::std::memory_order_release);
}

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    test<local_memory_selector>();
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    test<shared_memory_selector>();
    test<global_memory_selector>();
#endif

  return 0;
}
