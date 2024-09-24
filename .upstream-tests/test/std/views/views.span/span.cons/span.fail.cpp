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
// UNSUPPORTED: nvrtc
// UNSUPPORTED: hipcc
// Todo (hipcc): line 32 does not emit the expected error message


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

template<class T, size_t extent, size_t otherExtent>
hip::std::span<T, extent> createImplicitSpan(hip::std::span<T, otherExtent> s) {
    return {s}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

void checkCV ()
{
//  hip::std::span<               int>   sp;
    hip::std::span<const          int>  csp;
    hip::std::span<      volatile int>  vsp;
    hip::std::span<const volatile int> cvsp;

//  hip::std::span<               int, 0>   sp0;
    hip::std::span<const          int, 0>  csp0;
    hip::std::span<      volatile int, 0>  vsp0;
    hip::std::span<const volatile int, 0> cvsp0;

//  Try to remove const and/or volatile (dynamic -> dynamic)
    {
    hip::std::span<               int> s1{ csp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<               int> s2{ vsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<               int> s3{cvsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}

    hip::std::span<const          int> s4{ vsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
    hip::std::span<const          int> s5{cvsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}

    hip::std::span<      volatile int> s6{ csp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int>'}}
    hip::std::span<      volatile int> s7{cvsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int>'}}
    }

//  Try to remove const and/or volatile (static -> static)
    {
    hip::std::span<               int, 0> s1{ csp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 0>'}}
    hip::std::span<               int, 0> s2{ vsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 0>'}}
    hip::std::span<               int, 0> s3{cvsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 0>'}}

    hip::std::span<const          int, 0> s4{ vsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int, 0>'}}
    hip::std::span<const          int, 0> s5{cvsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int, 0>'}}

    hip::std::span<      volatile int, 0> s6{ csp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int, 0>'}}
    hip::std::span<      volatile int, 0> s7{cvsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int, 0>'}}
    }

//  Try to remove const and/or volatile (static -> dynamic)
    {
    hip::std::span<               int> s1{ csp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<               int> s2{ vsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<               int> s3{cvsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}

    hip::std::span<const          int> s4{ vsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
    hip::std::span<const          int> s5{cvsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}

    hip::std::span<      volatile int> s6{ csp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int>'}}
    hip::std::span<      volatile int> s7{cvsp0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int>'}}
    }

//  Try to remove const and/or volatile (static -> static)
    {
    hip::std::span<               int, 0> s1{ csp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 0>'}}
    hip::std::span<               int, 0> s2{ vsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 0>'}}
    hip::std::span<               int, 0> s3{cvsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 0>'}}

    hip::std::span<const          int, 0> s4{ vsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int, 0>'}}
    hip::std::span<const          int, 0> s5{cvsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int, 0>'}}

    hip::std::span<      volatile int, 0> s6{ csp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int, 0>'}}
    hip::std::span<      volatile int, 0> s7{cvsp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int, 0>'}}
    }
}

int main(int, char**)
{
    hip::std::span<int>      sp;
    hip::std::span<int, 0>   sp0;

    hip::std::span<float> s1{sp};    // expected-error {{no matching constructor for initialization of 'hip::std::span<float>'}}
    hip::std::span<float> s2{sp0};   // expected-error {{no matching constructor for initialization of 'hip::std::span<float>'}}
    hip::std::span<float, 0> s3{sp}; // expected-error {{no matching constructor for initialization of 'hip::std::span<float, 0>'}}
    hip::std::span<float, 0> s4{sp0};    // expected-error {{no matching constructor for initialization of 'hip::std::span<float, 0>'}}

    checkCV();

    // explicit constructor necessary
    {
    createImplicitSpan<int, 1>(sp);
    }

  return 0;
}
