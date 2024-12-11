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

#ifndef ALLOC_FIRST_H
#define ALLOC_FIRST_H

#include <hip/std/cassert>

#include "allocators.h"

struct alloc_first
{
    STATIC_MEMBER_VAR(allocator_constructed, bool);

    typedef A1<int> allocator_type;

    int data_;

    __host__ __device__ alloc_first() : data_(0) {}
    __host__ __device__ alloc_first(int d) : data_(d) {}
    __host__ __device__ alloc_first(hip::std::allocator_arg_t, const A1<int>& a)
        : data_(0)
    {
        assert(a.id() == 5);
        allocator_constructed() = true;
    }

    __host__ __device__ alloc_first(hip::std::allocator_arg_t, const A1<int>& a, int d)
        : data_(d)
    {
        assert(a.id() == 5);
        allocator_constructed() = true;
    }

    __host__ __device__ alloc_first(hip::std::allocator_arg_t, const A1<int>& a, const alloc_first& d)
        : data_(d.data_)
    {
        assert(a.id() == 5);
        allocator_constructed() = true;
    }

    __host__ __device__ ~alloc_first() {data_ = -1;}

    __host__ __device__ friend bool operator==(const alloc_first& x, const alloc_first& y)
        {return x.data_ == y.data_;}
    __host__ __device__ friend bool operator< (const alloc_first& x, const alloc_first& y)
        {return x.data_ < y.data_;}
};

#endif  // ALLOC_FIRST_H
