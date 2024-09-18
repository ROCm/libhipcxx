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

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Tuple, __tuple_convertible<Tuple, tuple> >
//   tuple(Tuple &&);
//
// template <class Tuple, __tuple_constructible<Tuple, tuple> >
//   tuple(Tuple &&);

// This test checks that we do not evaluate __make_tuple_types
// on the array.

// hip::std::array not supported
// #include <hip/std/array>
#include <hip/std/tuple>

#include "test_macros.h"

// hip::std::array not supported
/*
// Use 1256 to try and blow the template instantiation depth for all compilers.
typedef hip::std::array<char, 1256> array_t;
typedef hip::std::tuple<array_t> tuple_t;
*/

int main(int, char**)
{
  /*
  array_t arr;
  tuple_t tup(arr);
  */
  return 0;
}
