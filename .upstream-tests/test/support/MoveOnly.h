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

#ifndef MOVEONLY_H
#define MOVEONLY_H

#include "test_macros.h"

#if TEST_STD_VER >= 11

#include <hip/std/cstddef>
// #include <functional>

class MoveOnly
{
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

    int data_;
public:
    __host__ __device__ MoveOnly(int data = 1) : data_(data) {}
    __host__ __device__ MoveOnly(MoveOnly&& x)
        : data_(x.data_) {x.data_ = 0;}
    __host__ __device__ MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

    __host__ __device__ int get() const {return data_;}

    __host__ __device__ bool operator==(const MoveOnly& x) const {return data_ == x.data_;}
    __host__ __device__ bool operator< (const MoveOnly& x) const {return data_ <  x.data_;}
    __host__ __device__ MoveOnly operator+(const MoveOnly& x) const { return MoveOnly{data_ + x.data_}; }
    __host__ __device__ MoveOnly operator*(const MoveOnly& x) const { return MoveOnly{data_ * x.data_}; }
};

/*
namespace std {
template <>
struct hash<MoveOnly>
{
    typedef MoveOnly argument_type;
    typedef size_t result_type;
    std::size_t operator()(const MoveOnly& x) const {return x.get();}
};
}
*/

#endif  // TEST_STD_VER >= 11

#endif  // MOVEONLY_H
