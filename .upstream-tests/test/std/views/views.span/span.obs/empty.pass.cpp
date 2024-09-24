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

// [[nodiscard]] constexpr bool empty() const noexcept;
//


#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

struct A{};
__device__ constexpr int iArr1[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};
__device__           int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{
    static_assert( noexcept(hip::std::span<int>   ().empty()), "");
    static_assert( noexcept(hip::std::span<int, 0>().empty()), "");


    static_assert( hip::std::span<int>().empty(),            "");
    static_assert( hip::std::span<long>().empty(),           "");
    static_assert( hip::std::span<double>().empty(),         "");
    static_assert( hip::std::span<A>().empty(),              "");

    static_assert( hip::std::span<int, 0>().empty(),         "");
    static_assert( hip::std::span<long, 0>().empty(),        "");
    static_assert( hip::std::span<double, 0>().empty(),      "");
    static_assert( hip::std::span<A, 0>().empty(),           "");

    static_assert(!hip::std::span<const int>(iArr1, 1).empty(), "");
    static_assert(!hip::std::span<const int>(iArr1, 2).empty(), "");
    static_assert(!hip::std::span<const int>(iArr1, 3).empty(), "");
    static_assert(!hip::std::span<const int>(iArr1, 4).empty(), "");
    static_assert(!hip::std::span<const int>(iArr1, 5).empty(), "");

    assert( (hip::std::span<int>().empty()           ));
    assert( (hip::std::span<long>().empty()          ));
    assert( (hip::std::span<double>().empty()        ));
    assert( (hip::std::span<A>().empty()             ));

    assert( (hip::std::span<int, 0>().empty()        ));
    assert( (hip::std::span<long, 0>().empty()       ));
    assert( (hip::std::span<double, 0>().empty()     ));
    assert( (hip::std::span<A, 0>().empty()          ));

    assert(!(hip::std::span<int, 1>(iArr2, 1).empty()));
    assert(!(hip::std::span<int, 2>(iArr2, 2).empty()));
    assert(!(hip::std::span<int, 3>(iArr2, 3).empty()));
    assert(!(hip::std::span<int, 4>(iArr2, 4).empty()));
    assert(!(hip::std::span<int, 5>(iArr2, 5).empty()));

  return 0;
}
