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

// UNSUPPORTED: c++98, c++03, c++11

// XFAIL: gcc-7 && !nvrtc
// GCC 5: Fails for C++11, passes for C++14.
// GCC 6: Fails for C++11, passes for C++14.
// GCC 7: Fails for C++11, fails for C++14.
// GCC 8: Fails for C++11, passes for C++14.

// XFAIL: msvc-19.0

// <cuda/std/functional>

// equal_to, not_equal_to, less, et al.

// Test that these types can be constructed w/o an initializer in a constexpr
// context. This is specifically testing gcc.gnu.org/PR83921

#include <hip/std/functional>
#include "test_macros.h"

template <class T>
__host__ __device__
constexpr bool test_constexpr_context() {
  hip::std::equal_to<T> eq;
  ((void)eq);
  hip::std::not_equal_to<T> neq;
  ((void)neq);
  hip::std::less<T> l;
  ((void)l);
  hip::std::less_equal<T> le;
  ((void)le);
  hip::std::greater<T> g;
  ((void)g);
  hip::std::greater_equal<T> ge;
  ((void)ge);
  return true;
}

static_assert(test_constexpr_context<int>(), "");
static_assert(test_constexpr_context<void>(), "");

int main(int, char**) {


  return 0;
}
