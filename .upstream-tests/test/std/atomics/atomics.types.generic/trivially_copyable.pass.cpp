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

// NOTE: atomic<> of a TriviallyCopyable class is wrongly rejected by older
// clang versions. It was fixed right before the llvm 3.5 release. See PR18097.
// XFAIL: apple-clang-6.0, clang-3.4, clang-3.3

// <cuda/std/atomic>

// template <class T>
// struct atomic
// {
//     bool is_lock_free() const volatile noexcept;
//     bool is_lock_free() const noexcept;
//     void store(T desr, memory_order m = memory_order_seq_cst) volatile noexcept;
//     void store(T desr, memory_order m = memory_order_seq_cst) noexcept;
//     T load(memory_order m = memory_order_seq_cst) const volatile noexcept;
//     T load(memory_order m = memory_order_seq_cst) const noexcept;
//     operator T() const volatile noexcept;
//     operator T() const noexcept;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) volatile noexcept;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) noexcept;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order s, memory_order f) volatile noexcept;
//     bool compare_exchange_weak(T& expc, T desr, memory_order s, memory_order f) noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) volatile noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) noexcept;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) volatile noexcept;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                 memory_order m = memory_order_seq_cst) volatile noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order m = memory_order_seq_cst) noexcept;
//
//     atomic() noexcept = default;
//     constexpr atomic(T desr) noexcept;
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     T operator=(T) volatile noexcept;
//     T operator=(T) noexcept;
// };

#include <hip/std/atomic>
#include <hip/std/cassert>
// #include <hip/std/thread> // for thread_id
// #include <hip/std/chrono> // for nanoseconds

#include "test_macros.h"

struct TriviallyCopyable {
    __host__ __device__
    TriviallyCopyable ( int i ) : i_(i) {}
    int i_;
};

template <class T>
__host__ __device__
void test ( T t ) {
    hip::std::atomic<T> t0(t);
    hip::std::atomic_ref<T> t1(t);
}

int main(int, char**)
{
    test(TriviallyCopyable(42));
    // test(hip::std::this_thread::get_id());
    // test(hip::std::chrono::nanoseconds(2));

  return 0;
}
