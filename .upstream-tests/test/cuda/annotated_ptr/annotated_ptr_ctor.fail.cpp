//===----------------------------------------------------------------------===//
//
// Part of libhip++ (derived from libcu++),
// the C++ Standard Library for your entire system,
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

#include "utils.h"

template<typename T, typename P>
__host__ __device__ __noinline__
void test_ctor() {
  // default ctor, cpy and cpy assignment
  hip::annotated_ptr<T, P> def;
  def = def;
  hip::annotated_ptr<T, P> other(def);

  // from ptr
  T* rp = nullptr;
  hip::annotated_ptr<T, P> a(rp);
  assert(!a);

  // cpy ctor & asign to cv
  hip::annotated_ptr<const T, P> c(def);
  hip::annotated_ptr<volatile T, P> d(def);
  hip::annotated_ptr<const volatile T, P> e(def);
  c = e; // FAIL
  d = d; // FAIL
}

template<typename T, typename P>
__host__ __device__ __noinline__
void test_global_ctor() {
  test_ctor<T, P>();
}

__host__ __device__ __noinline__
void test_global_ctors() {
  test_global_ctor<int, hip::access_property::normal>();
}

int main(int argc, char ** argv)
{
  test_global_ctors();
  return 0;
}
