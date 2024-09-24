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

#include <hip/std/atomic>

template <class A, class T>
__host__ __device__
bool cmpxchg_weak_loop(A& atomic, T& expected, T desired) {
  for (int i = 0; i < 10; i++) {
    if (atomic.compare_exchange_weak(expected, desired) == true) {
      return true;
    }
  }

  return false;
}

template <class A, class T>
__host__ __device__
bool cmpxchg_weak_loop(A& atomic, T& expected, T desired,
                       hip::std::memory_order success,
                       hip::std::memory_order failure) {
  for (int i = 0; i < 10; i++) {
    if (atomic.compare_exchange_weak(expected, desired, success,
                                     failure) == true) {
      return true;
    }
  }

  return false;
}

template <class A, class T>
__host__ __device__
bool c_cmpxchg_weak_loop(A* atomic, T* expected, T desired) {
  for (int i = 0; i < 10; i++) {
    if (hip::std::atomic_compare_exchange_weak(atomic, expected, desired) == true) {
      return true;
    }
  }

  return false;
}

template <class A, class T>
__host__ __device__
bool c_cmpxchg_weak_loop(A* atomic, T* expected, T desired,
                         hip::std::memory_order success,
                         hip::std::memory_order failure) {
  for (int i = 0; i < 10; i++) {
    if (hip::std::atomic_compare_exchange_weak_explicit(atomic, expected, desired,
                                                   success, failure) == true) {
      return true;
    }
  }

  return false;
}
