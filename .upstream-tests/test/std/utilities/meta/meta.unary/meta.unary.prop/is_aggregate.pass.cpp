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

// <cuda/std/type_traits>

// template <class T> struct is_aggregate;
// template <class T> constexpr bool is_aggregate_v = is_aggregate<T>::value;

#include <hip/std/type_traits>
#include "test_macros.h"

template <class T>
__host__ __device__
void test_true()
{
#if defined(_LIBCUDACXX_IS_AGGREGATE)
    static_assert( hip::std::is_aggregate<T>::value, "");
    static_assert( hip::std::is_aggregate<const T>::value, "");
    static_assert( hip::std::is_aggregate<volatile T>::value, "");
    static_assert( hip::std::is_aggregate<const volatile T>::value, "");
    static_assert( hip::std::is_aggregate_v<T>, "");
    static_assert( hip::std::is_aggregate_v<const T>, "");
    static_assert( hip::std::is_aggregate_v<volatile T>, "");
    static_assert( hip::std::is_aggregate_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_false()
{
#if defined(_LIBCUDACXX_IS_AGGREGATE)
    static_assert(!hip::std::is_aggregate<T>::value, "");
    static_assert(!hip::std::is_aggregate<const T>::value, "");
    static_assert(!hip::std::is_aggregate<volatile T>::value, "");
    static_assert(!hip::std::is_aggregate<const volatile T>::value, "");
    static_assert(!hip::std::is_aggregate_v<T>, "");
    static_assert(!hip::std::is_aggregate_v<const T>, "");
    static_assert(!hip::std::is_aggregate_v<volatile T>, "");
    static_assert(!hip::std::is_aggregate_v<const volatile T>, "");
#endif
}

struct Aggregate {};
struct HasCons { __host__ __device__ HasCons(int); };
struct HasPriv {
  __host__ __device__
  void PreventUnusedPrivateMemberWarning();
private:
  int x;
};
struct Union { int x; void* y; };


int main(int, char**)
{
  {
    test_false<void>();
    test_false<int>();
    test_false<void*>();
    test_false<void()>();
    test_false<void() const>();
    test_false<void(Aggregate::*)(int) const>();
    test_false<Aggregate&>();
    test_false<HasCons>();
    test_false<HasPriv>();
  }
  {
    test_true<Aggregate>();
    test_true<Aggregate[]>();
    test_true<Aggregate[42][101]>();
    test_true<Union>();
  }

  return 0;
}
