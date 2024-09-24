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
//  ... test case crashes clang.

// <cuda/std/atomic>

// template <class T>
// struct atomic<T*>
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(T* desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(T* desr, memory_order m = memory_order_seq_cst);
//     T* load(memory_order m = memory_order_seq_cst) const volatile;
//     T* load(memory_order m = memory_order_seq_cst) const;
//     operator T*() const volatile;
//     operator T*() const;
//     T* exchange(T* desr, memory_order m = memory_order_seq_cst) volatile;
//     T* exchange(T* desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order s, memory_order f);
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                  memory_order m = memory_order_seq_cst);
//     T* fetch_add(ptrdiff_t op, memory_order m = memory_order_seq_cst) volatile;
//     T* fetch_add(ptrdiff_t op, memory_order m = memory_order_seq_cst);
//     T* fetch_sub(ptrdiff_t op, memory_order m = memory_order_seq_cst) volatile;
//     T* fetch_sub(ptrdiff_t op, memory_order m = memory_order_seq_cst);
//
//     atomic() = default;
//     constexpr atomic(T* desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//
//     T* operator=(T*) volatile;
//     T* operator=(T*);
//     T* operator++(int) volatile;
//     T* operator++(int);
//     T* operator--(int) volatile;
//     T* operator--(int);
//     T* operator++() volatile;
//     T* operator++();
//     T* operator--() volatile;
//     T* operator--();
//     T* operator+=(ptrdiff_t op) volatile;
//     T* operator+=(ptrdiff_t op);
//     T* operator-=(ptrdiff_t op) volatile;
//     T* operator-=(ptrdiff_t op);
// };

#include <hip/std/atomic>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include <cmpxchg_loop.h>

#include "test_macros.h"
#if !defined(TEST_COMPILER_C1XX)
  #include "placement_new.h"
#endif
#include "cuda_space_selector.h"

template <class A, class T, template<typename, typename> class Selector>
__host__ __device__
void
do_test()
{
    typedef typename hip::std::remove_pointer<T>::type X;
    Selector<A, constructor_initializer> sel;
    A & obj = *sel.construct(T(0));
    bool b0 = obj.is_lock_free();
    ((void)b0); // mark as unused
    assert(obj == T(0));
    obj.store(T(0));
    assert(obj == T(0));
    obj.store(T(1), hip::std::memory_order_release);
    assert(obj == T(1));
    assert(obj.load() == T(1));
    assert(obj.load(hip::std::memory_order_acquire) == T(1));
    assert(obj.exchange(T(2)) == T(1));
    assert(obj == T(2));
    assert(obj.exchange(T(3), hip::std::memory_order_relaxed) == T(2));
    assert(obj == T(3));
    T x = obj;
    assert(cmpxchg_weak_loop(obj, x, T(2)) == true);
    assert(obj == T(2));
    assert(x == T(3));
    assert(obj.compare_exchange_weak(x, T(1)) == false);
    assert(obj == T(2));
    assert(x == T(2));
    x = T(2);
    assert(obj.compare_exchange_strong(x, T(1)) == true);
    assert(obj == T(1));
    assert(x == T(2));
    assert(obj.compare_exchange_strong(x, T(0)) == false);
    assert(obj == T(1));
    assert(x == T(1));
    assert((obj = T(0)) == T(0));
    assert(obj == T(0));
    obj = T(2*sizeof(X));
    assert((obj += hip::std::ptrdiff_t(3)) == T(5*sizeof(X)));
    assert(obj == T(5*sizeof(X)));
    assert((obj -= hip::std::ptrdiff_t(3)) == T(2*sizeof(X)));
    assert(obj == T(2*sizeof(X)));
#if __cplusplus > 201703L
    {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
        TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {23};
        A& zero = *new (storage) A();
        assert(zero == T(0));
        zero.~A();
#endif
    }
#endif
}

template <class A, class T, template<typename, typename> class Selector>
__host__ __device__
void do_test_std()
{
    Selector<A, constructor_initializer> sel;
    A & obj = *sel.construct(nullptr);
    hip::std::atomic_init(&obj, T(1));
    assert(obj == T(1));
    hip::std::atomic_init(&obj, T(2));
    assert(obj == T(2));

    do_test<A, T, Selector>();
}

template <class A, class T, template<typename, typename> class Selector>
__host__ __device__
void test()
{
    do_test<A, T, Selector>();
    do_test<volatile A, T, Selector>();
}

template <class A, class T, template<typename, typename> class Selector>
__host__ __device__
void test_std()
{
    do_test_std<A, T, Selector>();
    do_test_std<volatile A, T, Selector>();
}

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    test_std<hip::std::atomic<int*>, int*, local_memory_selector>();
    test<hip::atomic<int*, hip::thread_scope_system>, int*, local_memory_selector>();
    test<hip::atomic<int*, hip::thread_scope_device>, int*, local_memory_selector>();
    test<hip::atomic<int*, hip::thread_scope_block>, int*, local_memory_selector>();
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    test_std<hip::std::atomic<int*>, int*, shared_memory_selector>();
    test<hip::atomic<int*, hip::thread_scope_system>, int*, shared_memory_selector>();
    test<hip::atomic<int*, hip::thread_scope_device>, int*, shared_memory_selector>();
    test<hip::atomic<int*, hip::thread_scope_block>, int*, shared_memory_selector>();

    // note: this _should_ be test_std, but for some reason that's resulting in an
    // unspecified launch failure, and I'm unsure what function is not __device__
    // and causes that to happen
    // the only difference is whether atomic_init is done or not, and that
    // _seems_ to be appropriately tested by the atomic_init test for hip::std::
    test<hip::std::atomic<int*>, int*, global_memory_selector>();
    test<hip::atomic<int*, hip::thread_scope_system>, int*, global_memory_selector>();
    test<hip::atomic<int*, hip::thread_scope_device>, int*, global_memory_selector>();
    test<hip::atomic<int*, hip::thread_scope_block>, int*, global_memory_selector>();
#endif
  return 0;
}
