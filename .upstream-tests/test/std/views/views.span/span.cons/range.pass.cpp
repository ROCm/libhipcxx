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

// <cuda/std/span>

//  template<class Container>
//    constexpr explicit(Extent != dynamic_extent) span(Container&);
//  template<class Container>
//    constexpr explicit(Extent != dynamic_extent) span(Container const&);

// This test checks for libc++'s non-conforming temporary extension to hip::std::span
// to support construction from containers that look like contiguous ranges.
//
// This extension is only supported when we don't ship <ranges>, and we can
// remove it once we get rid of _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES.

#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

//  Look ma - I'm a container!
template <typename T>
struct IsAContainer {
    __host__ __device__ constexpr IsAContainer() : v_{} {}
    __host__ __device__ constexpr size_t size() const {return 1;}
    __host__ __device__ constexpr       T *data() {return &v_;}
    __host__ __device__ constexpr const T *data() const {return &v_;}
    __host__ __device__ constexpr       T *begin() {return &v_;}
    __host__ __device__ constexpr const T *begin() const {return &v_;}
    __host__ __device__ constexpr       T *end() {return &v_ + 1;}
    __host__ __device__ constexpr const T *end() const {return &v_ + 1;}

    __host__ __device__ constexpr T const *getV() const {return &v_;} // for checking
    T v_;
};

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

__host__ __device__ void checkCV()
{
    IsAContainer<int> v{};

//  Types the same
    {
    hip::std::span<               int> s1{v};    // a span<               int> pointing at int.
    unused(s1);
    }

//  types different
    {
    hip::std::span<const          int> s1{v};    // a span<const          int> pointing at int.
    hip::std::span<      volatile int> s2{v};    // a span<      volatile int> pointing at int.
    hip::std::span<      volatile int> s3{v};    // a span<      volatile int> pointing at const int.
    hip::std::span<const volatile int> s4{v};    // a span<const volatile int> pointing at int.
    unused(s1);
    unused(s2);
    unused(s3);
    unused(s4);
    }

//  Constructing a const view from a temporary
    {
    hip::std::span<const int>    s1{IsAContainer<int>()};
    unused(s1);
    }
}


template <typename T>
__host__ __device__ constexpr bool testConstexprSpan()
{
    constexpr IsAContainer<const T> val{};
    hip::std::span<const T> s1{val};
    return s1.data() == val.getV() && s1.size() == 1;
}

template <typename T>
__host__ __device__ constexpr bool testConstexprSpanStatic()
{
    constexpr IsAContainer<const T> val{};
    hip::std::span<const T, 1> s1{val};
    return s1.data() == val.getV() && s1.size() == 1;
}

template <typename T>
__host__ __device__ void testRuntimeSpan()
{
    IsAContainer<T> val{};
    const IsAContainer<T> cVal;
    hip::std::span<T>       s1{val};
    hip::std::span<const T> s2{cVal};
    assert(s1.data() == val.getV()  && s1.size() == 1);
    assert(s2.data() == cVal.getV() && s2.size() == 1);
}

template <typename T>
__host__ __device__ void testRuntimeSpanStatic()
{
    IsAContainer<T> val{};
    const IsAContainer<T> cVal;
    hip::std::span<T, 1>       s1{val};
    hip::std::span<const T, 1> s2{cVal};
    assert(s1.data() == val.getV()  && s1.size() == 1);
    assert(s2.data() == cVal.getV() && s2.size() == 1);
}

struct A{};

int main(int, char**)
{
    static_assert(testConstexprSpan<int>(),    "");
    static_assert(testConstexprSpan<long>(),   "");
    static_assert(testConstexprSpan<double>(), "");
    static_assert(testConstexprSpan<A>(),      "");

    static_assert(testConstexprSpanStatic<int>(),    "");
    static_assert(testConstexprSpanStatic<long>(),   "");
    static_assert(testConstexprSpanStatic<double>(), "");
    static_assert(testConstexprSpanStatic<A>(),      "");

    testRuntimeSpan<int>();
    testRuntimeSpan<long>();
    testRuntimeSpan<double>();
    testRuntimeSpan<A>();

    testRuntimeSpanStatic<int>();
    testRuntimeSpanStatic<long>();
    testRuntimeSpanStatic<double>();
    testRuntimeSpanStatic<A>();

    checkCV();

    return 0;
}
