//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc
// UNSUPPORTED: c++98, c++03

#include "utils.h"

template <typename T, typename U>
__device__ __host__ __noinline__
void shared_mem_test_dev() {
  T* smem = alloc<T, 128>(true);
  smem[10] = 42;

  hip::annotated_ptr<U, hip::access_property::shared> p{smem + 10};

  assert(*p == 42);
}

__device__ __host__ __noinline__
void all_tests() {
  shared_mem_test_dev<int, int>();
  shared_mem_test_dev<int, const int>();
  shared_mem_test_dev<int, volatile int>();
  shared_mem_test_dev<int, const volatile int>();
}

__global__
void shared_mem_test() {
  all_tests();
};

// TODO: is this needed?
__device__ __host__ __noinline__
void test_all() {
#ifdef __CUDA_ARCH__
  all_tests();
#else
  shared_mem_test<<<1, 1, 0, 0>>>();
  assert_rt(cudaStreamSynchronize(0));
#endif
}

int main(int argc, char ** argv)
{
  test_all();
  return 0;
}
