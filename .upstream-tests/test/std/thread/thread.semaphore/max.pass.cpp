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
// UNSUPPORTED: hipcc
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/std/semaphore>

#include <hip/std/semaphore>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(hip::std::counting_semaphore<>::max() > 0, "");
  static_assert(hip::std::counting_semaphore<1>::max() >= 1, "");
  static_assert(hip::std::counting_semaphore<hip::std::numeric_limits<int>::max()>::max() >= 1, "");
  static_assert(hip::std::counting_semaphore<hip::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(hip::std::counting_semaphore<hip::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(hip::std::counting_semaphore<1>::max() == hip::std::binary_semaphore::max(), "");

  static_assert(hip::counting_semaphore<hip::thread_scope_system>::max() > 0, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_system, 1>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_system, hip::std::numeric_limits<int>::max()>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_system, hip::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_system, hip::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_system, 1>::max() == hip::binary_semaphore<hip::thread_scope_system>::max(), "");

  static_assert(hip::counting_semaphore<hip::thread_scope_device>::max() > 0, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_device, 1>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_device, hip::std::numeric_limits<int>::max()>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_device, hip::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_device, hip::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_device, 1>::max() == hip::binary_semaphore<hip::thread_scope_device>::max(), "");

  static_assert(hip::counting_semaphore<hip::thread_scope_block>::max() > 0, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_block, 1>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_block, hip::std::numeric_limits<int>::max()>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_block, hip::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_block, hip::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(hip::counting_semaphore<hip::thread_scope_block, 1>::max() == hip::binary_semaphore<hip::thread_scope_block>::max(), "");

  return 0;
}
