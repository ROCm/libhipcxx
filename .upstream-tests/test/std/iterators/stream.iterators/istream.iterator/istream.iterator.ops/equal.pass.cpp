//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// <cuda/std/iterator>

// class istream_iterator

// template <class T, class charT, class traits, class Distance>
//   bool operator==(const istream_iterator<T,charT,traits,Distance> &x,
//                   const istream_iterator<T,charT,traits,Distance> &y);
//
// template <class T, class charT, class traits, class Distance>
//   bool operator!=(const istream_iterator<T,charT,traits,Distance> &x,
//                   const istream_iterator<T,charT,traits,Distance> &y);

#include <hip/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#include <hip/std/sstream>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    hip::std::istringstream inf1(" 1 23");
    hip::std::istringstream inf2(" 1 23");
    hip::std::istream_iterator<int> i1(inf1);
    hip::std::istream_iterator<int> i2(inf1);
    hip::std::istream_iterator<int> i3(inf2);
    hip::std::istream_iterator<int> i4;
    hip::std::istream_iterator<int> i5;
    assert(i1 == i1);
    assert(i1 == i2);
    assert(i1 != i3);
    assert(i1 != i4);
    assert(i1 != i5);

    assert(i2 == i2);
    assert(i2 != i3);
    assert(i2 != i4);
    assert(i2 != i5);

    assert(i3 == i3);
    assert(i3 != i4);
    assert(i3 != i5);

    assert(i4 == i4);
    assert(i4 == i5);

    assert(hip::std::operator==(i1, i2));
    assert(hip::std::operator!=(i1, i3));

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
