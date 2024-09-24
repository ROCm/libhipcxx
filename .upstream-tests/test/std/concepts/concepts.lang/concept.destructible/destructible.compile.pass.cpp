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

// template<class T>
// concept destructible = is_nothrow_destructible_v<T>;

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include <hip/std/concepts>
#include <hip/std/type_traits>

struct Empty {};

struct Defaulted {
  ~Defaulted() = default;
};
struct Deleted {
  ~Deleted() = delete;
};

struct Noexcept {
  __host__ __device__ ~Noexcept() noexcept;
};
struct NoexceptTrue {
  __host__ __device__ ~NoexceptTrue() noexcept(true);
};
struct NoexceptFalse {
  __host__ __device__ ~NoexceptFalse() noexcept(false);
};

struct Protected {
protected:
  ~Protected() = default;
};
struct Private {
private:
  ~Private() = default;
};

template <class T>
struct NoexceptDependant {
  __host__ __device__ ~NoexceptDependant() noexcept(hip::std::is_same_v<T, int>);
};

template <class T>
__host__ __device__ void test() {
  static_assert(hip::std::destructible<T> == hip::std::is_nothrow_destructible_v<T>, "");
}

__host__ __device__ void test() {
  test<Empty>();

  test<Defaulted>();
  test<Deleted>();

  test<Noexcept>();
  test<NoexceptTrue>();
  test<NoexceptFalse>();

  test<Protected>();
  test<Private>();

  test<NoexceptDependant<int> >();
  test<NoexceptDependant<double> >();

  test<bool>();
  test<char>();
  test<int>();
  test<double>();
}

// Required for MSVC internal test runner compatibility.
int main(int, char**) { return 0; }
