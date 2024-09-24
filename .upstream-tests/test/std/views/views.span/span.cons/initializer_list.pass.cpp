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

#include <hip/std/span>
#include <hip/std/cassert>
#include <hip/std/cstddef>

#include "test_macros.h"
struct Sink {
    constexpr Sink() = default;
    __host__ __device__
    constexpr Sink(Sink*) {}
};

__host__ __device__
constexpr hip::std::size_t count(hip::std::span<const Sink> sp) {
    return sp.size();
}

template<int N>
__host__ __device__
constexpr hip::std::size_t countn(hip::std::span<const Sink, N> sp) {
    return sp.size();
}

__host__ __device__
constexpr bool test() {
    Sink a[10] = {};
    assert(count({a}) == 10);
    assert(count({a, a+10}) == 10);
    assert(countn<10>({a}) == 10);
    return true;
}

int main(int, char**) {
    test();
    static_assert(test(), "");

    return 0;
}
