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
// UNSUPPORTED: windows && (c++11 || c++14 || c++17)

// template<class F, class... Args>
// concept strict_weak_order;

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include <hip/std/concepts>

using hip::std::strict_weak_order;

static_assert(strict_weak_order<bool(int, int), int, int>, "");
static_assert(strict_weak_order<bool(int, int), double, double>, "");
static_assert(strict_weak_order<bool(int, double), double, double>, "");

static_assert(!strict_weak_order<bool (*)(), int, double>, "");
static_assert(!strict_weak_order<bool (*)(int), int, double>, "");
static_assert(!strict_weak_order<bool (*)(double), int, double>, "");

static_assert(!strict_weak_order<bool(double, double*), double, double*>, "");
static_assert(!strict_weak_order<bool(int&, int&), double&, double&>, "");

struct S1 {};
static_assert(strict_weak_order<bool (S1::*)(S1*), S1*, S1*>, "");
static_assert(strict_weak_order<bool (S1::*)(S1&), S1&, S1&>, "");

struct S2 {};

struct P1 {
  __host__ __device__ bool operator()(S1, S1) const;
};
static_assert(strict_weak_order<P1, S1, S1>, "");

struct P2 {
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
};
static_assert(!strict_weak_order<P2, S1, S2>, "");

struct P3 {
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
  __host__ __device__ bool operator()(S2, S1) const;
};
static_assert(!strict_weak_order<P3, S1, S2>, "");

struct P4 {
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
  __host__ __device__ bool operator()(S2, S1) const;
  __host__ __device__ bool operator()(S2, S2) const;
};
static_assert(strict_weak_order<P4, S1, S2>, "");

int main(int, char**) { return 0; }
