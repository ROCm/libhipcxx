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

// template <class ElementType, size_t Extent>
//     span<byte,
//          Extent == dynamic_extent
//              ? dynamic_extent
//              : sizeof(ElementType) * Extent>
//     as_writable_bytes(span<ElementType, Extent> s) noexcept;

#include <hip/std/span>

#include "test_macros.h"

__device__ constexpr int iArr2[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};

struct A {};

__host__ __device__ void f() {
  hip::std::as_writable_bytes(hip::std::span<const int>());            // expected-error {{no matching function for call to 'as_writable_bytes'}}
  hip::std::as_writable_bytes(hip::std::span<const long>());           // expected-error {{no matching function for call to 'as_writable_bytes'}}
  hip::std::as_writable_bytes(hip::std::span<const double>());         // expected-error {{no matching function for call to 'as_writable_bytes'}}
  hip::std::as_writable_bytes(hip::std::span<const A>());              // expected-error {{no matching function for call to 'as_writable_bytes'}}

  hip::std::as_writable_bytes(hip::std::span<const int, 0>());         // expected-error {{no matching function for call to 'as_writable_bytes'}}
  hip::std::as_writable_bytes(hip::std::span<const long, 0>());        // expected-error {{no matching function for call to 'as_writable_bytes'}}
  hip::std::as_writable_bytes(hip::std::span<const double, 0>());      // expected-error {{no matching function for call to 'as_writable_bytes'}}
  hip::std::as_writable_bytes(hip::std::span<const A, 0>());           // expected-error {{no matching function for call to 'as_writable_bytes'}}

  hip::std::as_writable_bytes(hip::std::span<const int>   (iArr2, 1));     // expected-error {{no matching function for call to 'as_writable_bytes'}}
  hip::std::as_writable_bytes(hip::std::span<const int, 1>(iArr2 + 5, 1)); // expected-error {{no matching function for call to 'as_writable_bytes'}}
}

int main(int, char**)
{
  return 0;
}
