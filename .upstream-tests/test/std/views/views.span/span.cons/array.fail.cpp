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

// <span>

// template<size_t N>
//     constexpr span(element_type (&arr)[N]) noexcept;
// template<size_t N>
//     constexpr span(array<value_type, N>& arr) noexcept;
// template<size_t N>
//     constexpr span(const array<value_type, N>& arr) noexcept;
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   — extent == dynamic_extent || N == extent is true, and
//   — remove_pointer_t<decltype(data(arr))>(*)[] is convertible to ElementType(*)[].
//


#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"

__device__                int   arr[] = {1,2,3};
__device__ const          int  carr[] = {4,5,6};
__device__       volatile int  varr[] = {7,8,9};
__device__ const volatile int cvarr[] = {1,3,5};

int main(int, char**)
{
//  Size wrong
    {
    hip::std::span<int, 2>   s1(arr); // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 2>'}}
    }

//  Type wrong
    {
    hip::std::span<float>    s1(arr);   // expected-error {{no matching constructor for initialization of 'hip::std::span<float>'}}
    hip::std::span<float, 3> s2(arr);   // expected-error {{no matching constructor for initialization of 'hip::std::span<float, 3>'}}
    }

//  CV wrong (dynamically sized)
    {
    hip::std::span<               int> s1{ carr};    // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<               int> s2{ varr};    // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<               int> s3{cvarr};    // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<const          int> s4{ varr};    // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
    hip::std::span<const          int> s5{cvarr};    // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
    hip::std::span<      volatile int> s6{ carr};    // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int>'}}
    hip::std::span<      volatile int> s7{cvarr};    // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int>'}}
    }

//  CV wrong (statically sized)
    {
    hip::std::span<               int,3> s1{ carr};  // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 3>'}}
    hip::std::span<               int,3> s2{ varr};  // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 3>'}}
    hip::std::span<               int,3> s3{cvarr};  // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 3>'}}
    hip::std::span<const          int,3> s4{ varr};  // expected-error {{no matching constructor for initialization of 'hip::std::span<const int, 3>'}}
    hip::std::span<const          int,3> s5{cvarr};  // expected-error {{no matching constructor for initialization of 'hip::std::span<const int, 3>'}}
    hip::std::span<      volatile int,3> s6{ carr};  // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int, 3>'}}
    hip::std::span<      volatile int,3> s7{cvarr};  // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int, 3>'}}
    }

  return 0;
}
