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

// template<class LHS, class RHS>
// concept assignable_from =
//   std::is_lvalue_reference_v<LHS> &&
//   std::common_reference_with<
//     const std::remove_reference_t<LHS>&,
//     const std::remove_reference_t<RHS>&> &&
//   requires (LHS lhs, RHS&& rhs) {
//     { lhs = std::forward<RHS>(rhs) } -> std::same_as<LHS>;
//   };

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include <hip/std/concepts>
#include <hip/std/type_traits>

#include "MoveOnly.h"

struct NoCommonRef {
  __host__ __device__ NoCommonRef& operator=(const int&);
};
static_assert(hip::std::is_assignable_v<NoCommonRef&, const int&>, "");
static_assert(!hip::std::assignable_from<NoCommonRef&, const int&>, ""); // no common reference type

struct Base {};
struct Derived : Base {};
static_assert(!hip::std::assignable_from<Base*, Derived*>, "");
static_assert( hip::std::assignable_from<Base*&, Derived*>, "");
static_assert( hip::std::assignable_from<Base*&, Derived*&>, "");
static_assert( hip::std::assignable_from<Base*&, Derived*&&>, "");
static_assert( hip::std::assignable_from<Base*&, Derived* const>, "");
static_assert( hip::std::assignable_from<Base*&, Derived* const&>, "");
static_assert( hip::std::assignable_from<Base*&, Derived* const&&>, "");
static_assert(!hip::std::assignable_from<Base*&, const Derived*>, "");
static_assert(!hip::std::assignable_from<Base*&, const Derived*&>, "");
static_assert(!hip::std::assignable_from<Base*&, const Derived*&&>, "");
static_assert(!hip::std::assignable_from<Base*&, const Derived* const>, "");
static_assert(!hip::std::assignable_from<Base*&, const Derived* const&>, "");
static_assert(!hip::std::assignable_from<Base*&, const Derived* const&&>, "");
static_assert( hip::std::assignable_from<const Base*&, Derived*>, "");
static_assert( hip::std::assignable_from<const Base*&, Derived*&>, "");
static_assert( hip::std::assignable_from<const Base*&, Derived*&&>, "");
static_assert( hip::std::assignable_from<const Base*&, Derived* const>, "");
static_assert( hip::std::assignable_from<const Base*&, Derived* const&>, "");
static_assert( hip::std::assignable_from<const Base*&, Derived* const&&>, "");
static_assert( hip::std::assignable_from<const Base*&, const Derived*>, "");
static_assert( hip::std::assignable_from<const Base*&, const Derived*&>, "");
static_assert( hip::std::assignable_from<const Base*&, const Derived*&&>, "");
static_assert( hip::std::assignable_from<const Base*&, const Derived* const>, "");
static_assert( hip::std::assignable_from<const Base*&, const Derived* const&>, "");
static_assert( hip::std::assignable_from<const Base*&, const Derived* const&&>, "");

struct VoidResultType {
    __host__ __device__ void operator=(const VoidResultType&);
};
static_assert(hip::std::is_assignable_v<VoidResultType&, const VoidResultType&>, "");
static_assert(!hip::std::assignable_from<VoidResultType&, const VoidResultType&>, "");

struct ValueResultType {
    __host__ __device__ ValueResultType operator=(const ValueResultType&);
};
static_assert(hip::std::is_assignable_v<ValueResultType&, const ValueResultType&>, "");
static_assert(!hip::std::assignable_from<ValueResultType&, const ValueResultType&>, "");

struct Locale {
    __host__ __device__ const Locale& operator=(const Locale&);
};
static_assert(hip::std::is_assignable_v<Locale&, const Locale&>, "");
static_assert(!hip::std::assignable_from<Locale&, const Locale&>, "");

struct Tuple {
    __host__ __device__ Tuple& operator=(const Tuple&);
    __host__ __device__ const Tuple& operator=(const Tuple&) const;
};
static_assert(!hip::std::assignable_from<Tuple, const Tuple&>, "");
static_assert( hip::std::assignable_from<Tuple&, const Tuple&>, "");
static_assert(!hip::std::assignable_from<Tuple&&, const Tuple&>, "");
static_assert(!hip::std::assignable_from<const Tuple, const Tuple&>, "");
static_assert( hip::std::assignable_from<const Tuple&, const Tuple&>, "");
static_assert(!hip::std::assignable_from<const Tuple&&, const Tuple&>, "");

// Finally, check a few simple cases.
static_assert( hip::std::assignable_from<int&, int>, "");
static_assert( hip::std::assignable_from<int&, int&>, "");
static_assert( hip::std::assignable_from<int&, int&&>, "");
static_assert(!hip::std::assignable_from<const int&, int>, "");
static_assert(!hip::std::assignable_from<const int&, int&>, "");
static_assert(!hip::std::assignable_from<const int&, int&&>, "");
static_assert( hip::std::assignable_from<volatile int&, int>, "");
static_assert( hip::std::assignable_from<volatile int&, int&>, "");
static_assert( hip::std::assignable_from<volatile int&, int&&>, "");
static_assert(!hip::std::assignable_from<int(&)[10], int>, "");
static_assert(!hip::std::assignable_from<int(&)[10], int(&)[10]>, "");
static_assert( hip::std::assignable_from<MoveOnly&, MoveOnly>, "");
static_assert(!hip::std::assignable_from<MoveOnly&, MoveOnly&>, "");
static_assert( hip::std::assignable_from<MoveOnly&, MoveOnly&&>, "");
static_assert(!hip::std::assignable_from<void, int>, "");
static_assert(!hip::std::assignable_from<void, void>, "");


int main(int, char**) { return 0; }
