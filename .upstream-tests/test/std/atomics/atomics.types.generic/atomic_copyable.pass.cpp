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

#include <hip/std/atomic>
#include <hip/std/utility>
#include <hip/std/cassert>
// #include <hip/std/thread> // for thread_id
// #include <hip/std/chrono> // for nanoseconds

#include "test_macros.h"
#include "cuda_space_selector.h"

template <class T>
__host__ __device__
void test_not_copy_constructible() {
  static_assert(!hip::std::is_constructible<T, T&&>(), "");
  static_assert(!hip::std::is_constructible<T, const T&>(), "");
  static_assert(!hip::std::is_assignable<T, T&&>(), "");
  static_assert(!hip::std::is_assignable<T, const T&>(), "");
}

template <class T>
__host__ __device__
void test_copy_constructible() {
  static_assert(hip::std::is_constructible<T, T&&>(), "");
  static_assert(hip::std::is_constructible<T, const T&>(), "");
  static_assert(!hip::std::is_assignable<T, T&&>(), "");
  static_assert(!hip::std::is_assignable<T, const T&>(), "");
}

template <class T, class A>
__host__ __device__
void test_atomic_ref_copy_ctor() {
  SHARED A val;
  val = 0;

  T t0(val);
  T t1(t0);

  t0++;
  t1++;

  assert(t1.load() == 2);
}

template <class T, class A>
__host__ __device__
void test_atomic_ref_move_ctor() {
  SHARED A val;
  val = 0;

  T t0(val);
  t0++;

  T t1(hip::std::move(t0));
  t1++;

  assert(t1.load() == 2);
}

int main(int, char**)
{
    test_not_copy_constructible<hip::std::atomic<int>>();
    test_not_copy_constructible<hip::atomic<int>>();

    test_copy_constructible<hip::std::atomic_ref<int>>();
    test_copy_constructible<hip::atomic_ref<int>>();

    test_atomic_ref_copy_ctor<hip::std::atomic_ref<int>, int>();
    test_atomic_ref_copy_ctor<hip::atomic_ref<int>, int>();
    test_atomic_ref_copy_ctor<const hip::std::atomic_ref<int>, int>();
    test_atomic_ref_copy_ctor<const hip::atomic_ref<int>, int>();

    test_atomic_ref_move_ctor<hip::std::atomic_ref<int>, int>();
    test_atomic_ref_move_ctor<hip::atomic_ref<int>, int>();
    test_atomic_ref_move_ctor<const hip::std::atomic_ref<int>, int>();
    test_atomic_ref_move_ctor<const hip::atomic_ref<int>, int>();
    // test(hip::std::this_thread::get_id());
    // test(hip::std::chrono::nanoseconds(2));

  return 0;
}
