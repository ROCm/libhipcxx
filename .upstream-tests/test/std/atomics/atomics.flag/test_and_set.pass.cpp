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
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// struct atomic_flag

// bool test_and_set(memory_order = memory_order_seq_cst);
// bool test_and_set(memory_order = memory_order_seq_cst) volatile;

#include <hip/std/atomic>
#include <hip/std/cassert>

#include "test_macros.h"
#include "cuda_space_selector.h"

template<template<typename, typename> class Selector>
__host__ __device__
void test()
{
    {
        Selector<hip::std::atomic_flag, default_initializer> sel;
        hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set() == 0);
        assert(f.test_and_set() == 1);
    }
    {
        Selector<hip::std::atomic_flag, default_initializer> sel;
        hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_relaxed) == 0);
        assert(f.test_and_set(hip::std::memory_order_relaxed) == 1);
    }
#ifndef __INTEL_COMPILER
    {
        Selector<hip::std::atomic_flag, default_initializer> sel;
        hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_consume) == 0);
        assert(f.test_and_set(hip::std::memory_order_consume) == 1);
    }
#endif
    {
        Selector<hip::std::atomic_flag, default_initializer> sel;
        hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_acquire) == 0);
        assert(f.test_and_set(hip::std::memory_order_acquire) == 1);
    }
    {
        Selector<hip::std::atomic_flag, default_initializer> sel;
        hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_release) == 0);
        assert(f.test_and_set(hip::std::memory_order_release) == 1);
    }
    {
        Selector<hip::std::atomic_flag, default_initializer> sel;
        hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_acq_rel) == 0);
        assert(f.test_and_set(hip::std::memory_order_acq_rel) == 1);
    }
    {
        Selector<hip::std::atomic_flag, default_initializer> sel;
        hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_seq_cst) == 0);
        assert(f.test_and_set(hip::std::memory_order_seq_cst) == 1);
    }
    {
        Selector<volatile hip::std::atomic_flag, default_initializer> sel;
        volatile hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set() == 0);
        assert(f.test_and_set() == 1);
    }
    {
        Selector<volatile hip::std::atomic_flag, default_initializer> sel;
        volatile hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_relaxed) == 0);
        assert(f.test_and_set(hip::std::memory_order_relaxed) == 1);
    }
#ifndef __INTEL_COMPILER
    {
        Selector<volatile hip::std::atomic_flag, default_initializer> sel;
        volatile hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_consume) == 0);
        assert(f.test_and_set(hip::std::memory_order_consume) == 1);
    }
#endif
    {
        Selector<volatile hip::std::atomic_flag, default_initializer> sel;
        volatile hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_acquire) == 0);
        assert(f.test_and_set(hip::std::memory_order_acquire) == 1);
    }
    {
        Selector<volatile hip::std::atomic_flag, default_initializer> sel;
        volatile hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_release) == 0);
        assert(f.test_and_set(hip::std::memory_order_release) == 1);
    }
    {
        Selector<volatile hip::std::atomic_flag, default_initializer> sel;
        volatile hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_acq_rel) == 0);
        assert(f.test_and_set(hip::std::memory_order_acq_rel) == 1);
    }
    {
        Selector<volatile hip::std::atomic_flag, default_initializer> sel;
        volatile hip::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(hip::std::memory_order_seq_cst) == 0);
        assert(f.test_and_set(hip::std::memory_order_seq_cst) == 1);
    }
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
