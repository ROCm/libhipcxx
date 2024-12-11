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

// nasty_containers is unsupported by nvrtc
// UNSUPPORTED: nvrtc

// template <BackInsertionContainer Cont>
//   front_insert_iterator<Cont>
//   front_inserter(Cont& x);

#include <hip/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#include <hip/std/list>
#endif
#include <hip/std/cassert>
#include "nasty_containers.h"

#include "test_macros.h"

template <class C>
__host__ __device__
void
test(C c)
{
    hip::std::front_insert_iterator<C> i = hip::std::front_inserter(c);
    i = 0;
    assert(c.size() == 1);
    assert(c.front() == 0);
}

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_LIST)
    test(hip::std::list<int>());
    test(nasty_list<int>());
#endif

  return 0;
}
