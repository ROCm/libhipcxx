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

// This test fails because diagnose_if doesn't emit all of the diagnostics
// when -fdelayed-template-parsing is enabled, like it is on Windows.
// XFAIL: LIBCXX-WINDOWS-FIXME

// REQUIRES: verify-support, diagnose-if-support
// UNSUPPORTED: libcpp-has-no-threads

// <cuda/std/atomic>

// Test that invalid memory order arguments are diagnosed where possible.

#include <hip/std/atomic>

int main(int, char**) {
    hip::std::atomic<int> x(42);
    volatile hip::std::atomic<int>& vx = x;
    int val1 = 1; ((void)val1);
    int val2 = 2; ((void)val2);
    // load operations
    {
        x.load(hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.load(hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.load(hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.load(hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        x.load(hip::std::memory_order_relaxed);
        x.load(hip::std::memory_order_consume);
        x.load(hip::std::memory_order_acquire);
        x.load(hip::std::memory_order_seq_cst);
    }
    {
        hip::std::atomic_load_explicit(&x, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_load_explicit(&x, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_load_explicit(&vx, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_load_explicit(&vx, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        hip::std::atomic_load_explicit(&x, hip::std::memory_order_relaxed);
        hip::std::atomic_load_explicit(&x, hip::std::memory_order_consume);
        hip::std::atomic_load_explicit(&x, hip::std::memory_order_acquire);
        hip::std::atomic_load_explicit(&x, hip::std::memory_order_seq_cst);
    }
    // store operations
    {
        x.store(42, hip::std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.store(42, hip::std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.store(42, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.store(42, hip::std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.store(42, hip::std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.store(42, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        x.store(42, hip::std::memory_order_relaxed);
        x.store(42, hip::std::memory_order_release);
        x.store(42, hip::std::memory_order_seq_cst);
    }
    {
        hip::std::atomic_store_explicit(&x, 42, hip::std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_store_explicit(&x, 42, hip::std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_store_explicit(&x, 42, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_store_explicit(&vx, 42, hip::std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_store_explicit(&vx, 42, hip::std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_store_explicit(&vx, 42, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        hip::std::atomic_store_explicit(&x, 42, hip::std::memory_order_relaxed);
        hip::std::atomic_store_explicit(&x, 42, hip::std::memory_order_release);
        hip::std::atomic_store_explicit(&x, 42, hip::std::memory_order_seq_cst);
    }
    // compare exchange weak
    {
        x.compare_exchange_weak(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.compare_exchange_weak(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.compare_exchange_weak(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.compare_exchange_weak(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        x.compare_exchange_weak(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_relaxed);
        x.compare_exchange_weak(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_consume);
        x.compare_exchange_weak(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acquire);
        x.compare_exchange_weak(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_seq_cst);
        // Test that the cmpxchg overload with only one memory order argument
        // does not generate any diagnostics.
        x.compare_exchange_weak(val1, val2, hip::std::memory_order_release);
    }
    {
        hip::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_compare_exchange_weak_explicit(&vx, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_compare_exchange_weak_explicit(&vx, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        hip::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_relaxed);
        hip::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_consume);
        hip::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acquire);
        hip::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_seq_cst);
    }
    // compare exchange strong
    {
        x.compare_exchange_strong(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.compare_exchange_strong(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.compare_exchange_strong(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.compare_exchange_strong(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        x.compare_exchange_strong(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_relaxed);
        x.compare_exchange_strong(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_consume);
        x.compare_exchange_strong(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acquire);
        x.compare_exchange_strong(val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_seq_cst);
        // Test that the cmpxchg overload with only one memory order argument
        // does not generate any diagnostics.
        x.compare_exchange_strong(val1, val2, hip::std::memory_order_release);
    }
    {
        hip::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_compare_exchange_strong_explicit(&vx, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        hip::std::atomic_compare_exchange_strong_explicit(&vx, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        hip::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_relaxed);
        hip::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_consume);
        hip::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_acquire);
        hip::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, hip::std::memory_order_seq_cst, hip::std::memory_order_seq_cst);
    }

  return 0;
}
