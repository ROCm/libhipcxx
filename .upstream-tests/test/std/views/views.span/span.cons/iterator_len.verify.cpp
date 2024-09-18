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

// template <class It>
// constexpr explicit(Extent != dynamic_extent) span(It first, size_type count);
//  If Extent is not equal to dynamic_extent, then count shall be equal to Extent.
//

#include <hip/std/span>
#include <hip/std/cstddef>

template <class T, size_t extent>
__host__ __device__
hip::std::span<T, extent> createImplicitSpan(T* ptr, size_t len) {
  return {ptr, len}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**) {
  // explicit constructor necessary
  int arr[] = {1, 2, 3};
  createImplicitSpan<int, 1>(arr, 3);

  hip::std::span<int> sp = {0, 0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int>'}}
  hip::std::span<int, 2> sp2 = {0, 0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<int, 2>'}}
  hip::std::span<const int> csp = {0, 0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int>'}}
  hip::std::span<const int, 2> csp2 = {0, 0}; // expected-error {{no matching constructor for initialization of 'hip::std::span<const int, 2>'}}

  return 0;
}
