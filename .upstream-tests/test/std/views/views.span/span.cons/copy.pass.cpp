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
// UNSUPPORTED: c++03, c++11

// <span>

//  constexpr span(const span& other) noexcept = default;

#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool doCopy(const T &rhs)
{
    ASSERT_NOEXCEPT(T{rhs});
    T lhs{rhs};
    return lhs.data() == rhs.data()
     &&    lhs.size() == rhs.size();
}

struct A{};

template <typename T>
__host__ __device__
void testCV ()
{
    int  arr[] = {1,2,3};
    assert((doCopy(hip::std::span<T>  ()          )));
    assert((doCopy(hip::std::span<T,0>()          )));
    assert((doCopy(hip::std::span<T>  (&arr[0], 1))));
    assert((doCopy(hip::std::span<T,1>(&arr[0], 1))));
    assert((doCopy(hip::std::span<T>  (&arr[0], 2))));
    assert((doCopy(hip::std::span<T,2>(&arr[0], 2))));
}

__device__ constexpr int carr[] = {1,2,3};

int main(int, char**)
{

    STATIC_ASSERT_CXX14(doCopy(hip::std::span<      int>  ()));
    STATIC_ASSERT_CXX14(doCopy(hip::std::span<      int,0>()));
    STATIC_ASSERT_CXX14(doCopy(hip::std::span<const int>  (&carr[0], 1)));
    STATIC_ASSERT_CXX14(doCopy(hip::std::span<const int,1>(&carr[0], 1)));
    STATIC_ASSERT_CXX14(doCopy(hip::std::span<const int>  (&carr[0], 2)));
    STATIC_ASSERT_CXX14(doCopy(hip::std::span<const int,2>(&carr[0], 2)));

    STATIC_ASSERT_CXX14(doCopy(hip::std::span<long>()));
    STATIC_ASSERT_CXX14(doCopy(hip::std::span<double>()));
    STATIC_ASSERT_CXX14(doCopy(hip::std::span<A>()));

    testCV<               int>();
    testCV<const          int>();
    testCV<      volatile int>();
    testCV<const volatile int>();

  return 0;
}
