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

// template<class OtherElementType, size_t OtherExtent>
//    constexpr span(const span<OtherElementType, OtherExtent>& s) noexcept;
//
//  Remarks: This constructor shall not participate in overload resolution unless:
//      Extent == dynamic_extent || Extent == OtherExtent is true, and
//      OtherElementType(*)[] is convertible to ElementType(*)[].


#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

__host__ __device__
void checkCV()
{
    hip::std::span<               int>   sp;
//  hip::std::span<const          int>  csp;
    hip::std::span<      volatile int>  vsp;
//  hip::std::span<const volatile int> cvsp;

    hip::std::span<               int, 0>   sp0;
//  hip::std::span<const          int, 0>  csp0;
    hip::std::span<      volatile int, 0>  vsp0;
//  hip::std::span<const volatile int, 0> cvsp0;

//  dynamic -> dynamic
    {
        hip::std::span<const          int> s1{  sp}; // a hip::std::span<const          int> pointing at int.
        hip::std::span<      volatile int> s2{  sp}; // a hip::std::span<      volatile int> pointing at int.
        hip::std::span<const volatile int> s3{  sp}; // a hip::std::span<const volatile int> pointing at int.
        hip::std::span<const volatile int> s4{ vsp}; // a hip::std::span<const volatile int> pointing at volatile int.
        assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }

//  static -> static
    {
        hip::std::span<const          int, 0> s1{  sp0}; // a hip::std::span<const          int> pointing at int.
        hip::std::span<      volatile int, 0> s2{  sp0}; // a hip::std::span<      volatile int> pointing at int.
        hip::std::span<const volatile int, 0> s3{  sp0}; // a hip::std::span<const volatile int> pointing at int.
        hip::std::span<const volatile int, 0> s4{ vsp0}; // a hip::std::span<const volatile int> pointing at volatile int.
        assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }

//  static -> dynamic
    {
        hip::std::span<const          int> s1{  sp0};    // a hip::std::span<const          int> pointing at int.
        hip::std::span<      volatile int> s2{  sp0};    // a hip::std::span<      volatile int> pointing at int.
        hip::std::span<const volatile int> s3{  sp0};    // a hip::std::span<const volatile int> pointing at int.
        hip::std::span<const volatile int> s4{ vsp0};    // a hip::std::span<const volatile int> pointing at volatile int.
        assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }

//  dynamic -> static (not allowed)
}


template <typename T>
__host__ __device__
constexpr bool testConstexprSpan()
{
    hip::std::span<T>    s0{};
    hip::std::span<T, 0> s1{};
    hip::std::span<T>    s2(s1); // static -> dynamic
    ASSERT_NOEXCEPT(hip::std::span<T>   {s0});
    ASSERT_NOEXCEPT(hip::std::span<T, 0>{s1});
    ASSERT_NOEXCEPT(hip::std::span<T>   {s1});

    return
        s1.data() == nullptr && s1.size() == 0
    &&  s2.data() == nullptr && s2.size() == 0;
}


template <typename T>
__host__ __device__
void testRuntimeSpan()
{
    hip::std::span<T>    s0{};
    hip::std::span<T, 0> s1{};
    hip::std::span<T>    s2(s1); // static -> dynamic
    ASSERT_NOEXCEPT(hip::std::span<T>   {s0});
    ASSERT_NOEXCEPT(hip::std::span<T, 0>{s1});
    ASSERT_NOEXCEPT(hip::std::span<T>   {s1});

    assert(s1.data() == nullptr && s1.size() == 0);
    assert(s2.data() == nullptr && s2.size() == 0);
}


struct A{};

int main(int, char**)
{
    STATIC_ASSERT_CXX14(testConstexprSpan<int>());
    STATIC_ASSERT_CXX14(testConstexprSpan<long>());
    STATIC_ASSERT_CXX14(testConstexprSpan<double>());
    STATIC_ASSERT_CXX14(testConstexprSpan<A>());

    testRuntimeSpan<int>();
    testRuntimeSpan<long>();
    testRuntimeSpan<double>();
    testRuntimeSpan<A>();

    checkCV();

  return 0;
}
