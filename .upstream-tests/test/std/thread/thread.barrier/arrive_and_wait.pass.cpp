//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

// <cuda/std/barrier>

#include <cuda/std/barrier>

#include "test_macros.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template<typename Barrier,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__
void test()
{
  Selector<Barrier, Initializer> sel;
  SHARED Barrier * b;
  b = sel.construct(2);

  auto worker = LAMBDA (){
    for(int i = 0; i < 10; ++i)
      b->arrive_and_wait();
  };

  concurrent_agents_launch(worker, worker);
}

int main(int, char**)
{
    NV_IF_ELSE_TARGET(NV_IS_HOST,(
        cuda_thread_count = 2;

        test<cuda::std::barrier<>, local_memory_selector>();
        test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>();
        test<cuda::barrier<cuda::thread_scope_device>, local_memory_selector>();
        test<cuda::barrier<cuda::thread_scope_system>, local_memory_selector>();
    ),(
        test<cuda::std::barrier<>, shared_memory_selector>();
        test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>();
        test<cuda::barrier<cuda::thread_scope_device>, shared_memory_selector>();
        test<cuda::barrier<cuda::thread_scope_system>, shared_memory_selector>();

        test<cuda::std::barrier<>, global_memory_selector>();
        test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>();
        test<cuda::barrier<cuda::thread_scope_device>, global_memory_selector>();
        test<cuda::barrier<cuda::thread_scope_system>, global_memory_selector>();
    ))

  return 0;
}
