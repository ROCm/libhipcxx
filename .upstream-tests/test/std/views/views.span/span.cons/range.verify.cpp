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

    __host__ __device__ constexpr const T *getV() const {return &v_;} // for checking
    T v_;
};

template <typename T>
struct NotAContainerNoData {
    __host__ __device__ size_t size() const {return 0;}
};

template <typename T>
struct NotAContainerNoSize {
    __host__ __device__ const T *data() const {return nullptr;}
};

template <typename T>
struct NotAContainerPrivate {
private:
    __host__ __device__ size_t size() const {return 0;}
    __host__ __device__ const T *data() const {return nullptr;}
};

template<class T, size_t extent, class container>
__host__ __device__ hip::std::span<T, extent> createImplicitSpan(container c) {
    return {c}; // expected-error-re {{no matching constructor for initialization of 'hip::std::span<{{.+}}>'}}
}

int main(int, char**)
{

//  Making non-const spans from const sources (a temporary binds to `const &`)
    {
    hip::std::span<int>    s1{IsAContainer<int>()};          // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    }

//  Missing size and/or data
    {
    hip::std::span<const int>    s1{NotAContainerNoData<int>()};   // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
    hip::std::span<const int>    s3{NotAContainerNoSize<int>()};   // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
    hip::std::span<const int>    s5{NotAContainerPrivate<int>()};  // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
    }

//  Not the same type
    {
    IsAContainer<int> c;
    hip::std::span<float>    s1{c};   // expected-error {{no matching constructor for initialization of 'hip::std::span<float>'}}
    }

//  CV wrong
    {
    IsAContainer<const          int> c;
    IsAContainer<const volatile int> cv;
    IsAContainer<      volatile int> v;

    hip::std::span<               int> s1{c};    // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<               int> s2{v};    // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<               int> s3{cv};   // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
    hip::std::span<const          int> s4{v};    // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
    hip::std::span<const          int> s5{cv};   // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
    hip::std::span<      volatile int> s6{c};    // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int>'}}
    hip::std::span<      volatile int> s7{cv};   // expected-error {{no matching constructor for initialization of 'hip::std::span<volatile int>'}}
    }

// explicit constructor necessary
    {
    IsAContainer<int> c;
    const IsAContainer<int> cc;

    createImplicitSpan<int, 1>(c);
    createImplicitSpan<int, 1>(cc);
    }

    return 0;
}
