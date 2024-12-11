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

// const T* operator->() const;

#include <hip/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#include <hip/std/sstream>
#include <hip/std/cassert>

#include "test_macros.h"

struct A
{
    double d_;
    int i_;
};

void operator&(A const&) {}

hip::std::istream& operator>>(hip::std::istream& is, A& a)
{
    return is >> a.d_ >> a.i_;
}

int main(int, char**)
{
    hip::std::istringstream inf("1.5  23 ");
    hip::std::istream_iterator<A> i(inf);
    assert(i->d_ == 1.5);
    assert(i->i_ == 23);

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
