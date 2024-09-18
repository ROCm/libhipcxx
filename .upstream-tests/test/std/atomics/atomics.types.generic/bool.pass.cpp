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

// template <class T>
// struct atomic
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(T desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(T desr, memory_order m = memory_order_seq_cst);
//     T load(memory_order m = memory_order_seq_cst) const volatile;
//     T load(memory_order m = memory_order_seq_cst) const;
//     operator T() const volatile;
//     operator T() const;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) volatile;
//     T exchange(T desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(T& expc, T desr, memory_order s, memory_order f);
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(T& expc, T desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order m = memory_order_seq_cst);
//
//     atomic() = default;
//     constexpr atomic(T desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     T operator=(T) volatile;
//     T operator=(T);
// };
//
// typedef atomic<bool> atomic_bool;

#include <hip/std/atomic>
#include <hip/std/cassert>

#include <cmpxchg_loop.h>

#include "test_macros.h"
#if !defined(TEST_COMPILER_C1XX)
  #include "placement_new.h"
#endif
#include "cuda_space_selector.h"

template<template<hip::thread_scope> typename Atomic, hip::thread_scope Scope, template<typename, typename> class Selector>
__host__ __device__ __noinline__
void do_test()
{
    {
        Selector<volatile Atomic<Scope>, constructor_initializer> sel;
        volatile Atomic<Scope> & obj = *sel.construct(true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        (void)b0; // to placate scan-build
        obj.store(false);
        assert(obj == false);
        obj.store(true, hip::std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(hip::std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, hip::std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(cmpxchg_weak_loop(obj, x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true,
                                         hip::std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        obj.store(true);
        x = true;
        assert(cmpxchg_weak_loop(obj, x, false,
                                 hip::std::memory_order_seq_cst,
                                 hip::std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_strong(x, true,
                                         hip::std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false,
                                           hip::std::memory_order_seq_cst,
                                           hip::std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
        assert((obj = true) == true);
        assert(obj == true);
    }
    {
        Selector<Atomic<Scope>, constructor_initializer> sel;
        Atomic<Scope> & obj = *sel.construct(true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        (void)b0; // to placate scan-build
        obj.store(false);
        assert(obj == false);
        obj.store(true, hip::std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(hip::std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, hip::std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(cmpxchg_weak_loop(obj, x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true,
                                         hip::std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        obj.store(true);
        x = true;
        assert(cmpxchg_weak_loop(obj, x, false,
                                 hip::std::memory_order_seq_cst,
                                 hip::std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_strong(x, true,
                                         hip::std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false,
                                           hip::std::memory_order_seq_cst,
                                           hip::std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
        assert((obj = true) == true);
        assert(obj == true);
    }
    {
        Selector<Atomic<Scope>, constructor_initializer> sel;
        Atomic<Scope> & obj = *sel.construct(true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        (void)b0; // to placate scan-build
        obj.store(false);
        assert(obj == false);
        obj.store(true, hip::std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(hip::std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, hip::std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(cmpxchg_weak_loop(obj, x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true,
                                         hip::std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        obj.store(true);
        x = true;
        assert(cmpxchg_weak_loop(obj, x, false,
                                 hip::std::memory_order_seq_cst,
                                 hip::std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_strong(x, true,
                                         hip::std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false,
                                           hip::std::memory_order_seq_cst,
                                           hip::std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
        assert((obj = true) == true);
        assert(obj == true);
    }
#if __cplusplus > 201703L
    {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
        typedef Atomic<Scope> A;
        TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {1};
        A& zero = *new (storage) A();
        assert(zero == false);
        zero.~A();
#endif
    }
#endif
}

template<hip::thread_scope Scope>
using cuda_std_atomic = hip::std::atomic<bool>;

template<hip::thread_scope Scope>
using cuda_atomic = hip::atomic<bool, Scope>;


int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    do_test<cuda_std_atomic, hip::thread_scope_system, local_memory_selector>();
    do_test<cuda_atomic, hip::thread_scope_system, local_memory_selector>();
    do_test<cuda_atomic, hip::thread_scope_device, local_memory_selector>();
    do_test<cuda_atomic, hip::thread_scope_block, local_memory_selector>();
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    do_test<cuda_std_atomic, hip::thread_scope_system, shared_memory_selector>();
    do_test<cuda_atomic, hip::thread_scope_system, shared_memory_selector>();
    do_test<cuda_atomic, hip::thread_scope_device, shared_memory_selector>();
    do_test<cuda_atomic, hip::thread_scope_block, shared_memory_selector>();

    do_test<cuda_std_atomic, hip::thread_scope_system, global_memory_selector>();
    do_test<cuda_atomic, hip::thread_scope_system, global_memory_selector>();
    do_test<cuda_atomic, hip::thread_scope_device, global_memory_selector>();
    do_test<cuda_atomic, hip::thread_scope_block, global_memory_selector>();
#endif

  return 0;
}
