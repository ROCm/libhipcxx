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
  {
    hip::annotated_ptr<T, P> temp;
    temp = def;
    unused(temp);
  }
  hip::annotated_ptr<T, P> other(def);
  unused(other);

  // from ptr
  T* rp = nullptr;
  hip::annotated_ptr<T, P> a(rp);
  assert(!a);

  // cpy ctor & asign to cv
  hip::annotated_ptr<const T, P> c(def);
  hip::annotated_ptr<volatile T, P> d(def);
  hip::annotated_ptr<const volatile T, P> e(def);
  c = def;
  d = def;
  e = def;

  // from c|v to c|v|cv
  hip::annotated_ptr<const T, P> f(c);
  hip::annotated_ptr<volatile T, P> g(d);
  hip::annotated_ptr<const volatile T, P> h(e);
  f = c;
  g = d;
  h = e;
  unused(f, g, h);

  // to cv
  hip::annotated_ptr<const volatile T, P> i(c);
  hip::annotated_ptr<const volatile T, P> j(d);
  i = c;
  j = d;
}

template<typename T, typename P>
__host__ __device__ __noinline__
void test_global_ctor() {
  test_ctor<T, P>();

  // from ptr + prop
  T* rp = nullptr;
  P p;
  hip::annotated_ptr<T, hip::access_property> a(rp, p);
  hip::annotated_ptr<const T, hip::access_property> b(rp, p);
  hip::annotated_ptr<volatile T, hip::access_property> c(rp, p);
  hip::annotated_ptr<const volatile T, hip::access_property> d(rp, p);
}

__host__ __device__ __noinline__
void test_global_ctors() {
  test_global_ctor<int, hip::access_property::normal>();
  test_global_ctor<int, hip::access_property::streaming>();
  test_global_ctor<int, hip::access_property::persisting>();
  test_global_ctor<int, hip::access_property::global>();
  test_global_ctor<int, hip::access_property>();
  test_ctor<int, hip::access_property::shared>();
}

int main(int argc, char ** argv)
{
  test_global_ctors();
  return 0;
}
