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

// Compiler bug for being unable to convert __nv_hdl lambdas
// XFAIL: msvc-19.33

// <cuda/std/barrier>

#include <hip/std/barrier>

#include "test_macros.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template<template<typename> typename Barrier,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__
void test()
{
  global_memory_selector<int> int_sel;
  SHARED int * x;
  x = int_sel.construct(0);

  auto comp = LAMBDA () { *x += 1; };

  Selector<Barrier<decltype(comp)>, Initializer> sel;
  SHARED Barrier<decltype(comp)> * b;
  b = sel.construct(2, comp);

  auto worker = LAMBDA () {
      for(int i = 0; i < 10; ++i)
        b->arrive_and_wait();
      assert(*x == 10);
  };

  concurrent_agents_launch(worker, worker);

  assert(*x == 10);
}

template<typename Comp>
using std_barrier = hip::std::barrier<Comp>;
template<typename Comp>
using block_barrier = hip::barrier<hip::thread_scope_block, Comp>;
template<typename Comp>
using device_barrier = hip::barrier<hip::thread_scope_device, Comp>;
template<typename Comp>
using system_barrier = hip::barrier<hip::thread_scope_system, Comp>;

int main(int, char**)
{
#ifndef __CUDA_ARCH__
  gpu_thread_count = 2;

  test<std_barrier, local_memory_selector>();
  test<block_barrier, local_memory_selector>();
  test<device_barrier, local_memory_selector>();
  test<system_barrier, local_memory_selector>();
#else
  test<std_barrier, shared_memory_selector>();
  test<block_barrier, shared_memory_selector>();
  test<device_barrier, shared_memory_selector>();
  test<system_barrier, shared_memory_selector>();

  test<std_barrier, global_memory_selector>();
  test<block_barrier, global_memory_selector>();
  test<device_barrier, global_memory_selector>();
  test<system_barrier, global_memory_selector>();
#endif

  return 0;
}