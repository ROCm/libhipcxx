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

// <cuda/std/memory>

// template <ObjectType T> T* addressof(T& r);

#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

struct A
{
    __host__ __device__ void operator&() const {}
};

struct nothing {
    __host__ __device__ operator char&()
    {
        static char c;
        return c;
    }
};

int main(int, char**)
{
    {
    int i;
    double d;
    assert(hip::std::addressof(i) == &i);
    assert(hip::std::addressof(d) == &d);
    A* tp = new A;
    const A* ctp = tp;
    assert(hip::std::addressof(*tp) == tp);
    assert(hip::std::addressof(*ctp) == tp);
    delete tp;
    }
    {
    union
    {
        nothing n;
        int i;
    };
    assert(hip::std::addressof(n) == (void*)hip::std::addressof(i));
    }

  return 0;
}
