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
// concept totally_ordered;

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#pragma clang diagnostic ignored "-Wordered-compare-function-pointers"
#endif

#include <hip/std/concepts>

#include <hip/std/array>

#include "test_macros.h"
#include "compare_types.h"

using hip::std::totally_ordered;

// `models_totally_ordered` checks that `totally_ordered` subsumes
// `std::equality_comparable`. This overload should *never* be called.
#if TEST_STD_VER > 17

template <hip::std::equality_comparable T>
__host__ __device__ constexpr bool models_totally_ordered() noexcept {
  return false;
}

template <hip::std::totally_ordered T>
__host__ __device__ constexpr bool models_totally_ordered() noexcept {
  return true;
}

#else

_LIBCUDACXX_TEMPLATE(class T)
  (requires hip::std::totally_ordered<T>)
__host__ __device__ constexpr bool models_totally_ordered() noexcept {
  return true;
}

#endif // TEST_STD_VER > 17

namespace fundamentals {
static_assert(models_totally_ordered<int>(), "");
static_assert(models_totally_ordered<double>(), "");
static_assert(models_totally_ordered<void*>(), "");
static_assert(models_totally_ordered<char*>(), "");
static_assert(models_totally_ordered<char const*>(), "");
static_assert(models_totally_ordered<char volatile*>(), "");
static_assert(models_totally_ordered<char const volatile*>(), "");
static_assert(models_totally_ordered<wchar_t&>(), "");
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
static_assert(models_totally_ordered<char8_t const&>(), "");
#endif // TEST_STD_VER > 17 && defined(__cpp_char8_t)
static_assert(models_totally_ordered<char16_t volatile&>(), "");
static_assert(models_totally_ordered<char32_t const volatile&>(), "");
static_assert(models_totally_ordered<unsigned char&&>(), "");
static_assert(models_totally_ordered<unsigned short const&&>(), "");
static_assert(models_totally_ordered<unsigned int volatile&&>(), "");
static_assert(models_totally_ordered<unsigned long const volatile&&>(), "");
static_assert(models_totally_ordered<int[5]>(), "");
static_assert(models_totally_ordered<int (*)(int)>(), "");
static_assert(models_totally_ordered<int (&)(int)>(), "");
static_assert(models_totally_ordered<int (*)(int) noexcept>(), "");
static_assert(models_totally_ordered<int (&)(int) noexcept>(), "");

#if !defined(TEST_COMPILER_GCC) && defined(INVESTIGATE_COMPILER_BUG)
static_assert(!totally_ordered<hip::std::nullptr_t>, "");
#endif

struct S {};
static_assert(!totally_ordered<S>, "");
static_assert(!totally_ordered<int S::*>, "");
static_assert(!totally_ordered<int (S::*)()>, "");
static_assert(!totally_ordered<int (S::*)() noexcept>, "");
static_assert(!totally_ordered<int (S::*)() &>, "");
static_assert(!totally_ordered<int (S::*)() & noexcept>, "");
static_assert(!totally_ordered<int (S::*)() &&>, "");
static_assert(!totally_ordered < int (S::*)() && noexcept >, "");
static_assert(!totally_ordered<int (S::*)() const>, "");
static_assert(!totally_ordered<int (S::*)() const noexcept>, "");
static_assert(!totally_ordered<int (S::*)() const&>, "");
static_assert(!totally_ordered<int (S::*)() const & noexcept>, "");
static_assert(!totally_ordered<int (S::*)() const&&>, "");
static_assert(!totally_ordered < int (S::*)() const&& noexcept >, "");
static_assert(!totally_ordered<int (S::*)() volatile>, "");
static_assert(!totally_ordered<int (S::*)() volatile noexcept>, "");
static_assert(!totally_ordered<int (S::*)() volatile&>, "");
static_assert(!totally_ordered<int (S::*)() volatile & noexcept>, "");
static_assert(!totally_ordered<int (S::*)() volatile&&>, "");
static_assert(!totally_ordered < int (S::*)() volatile&& noexcept >, "");
static_assert(!totally_ordered<int (S::*)() const volatile>, "");
static_assert(!totally_ordered<int (S::*)() const volatile noexcept>, "");
static_assert(!totally_ordered<int (S::*)() const volatile&>, "");
static_assert(!totally_ordered<int (S::*)() const volatile & noexcept>, "");
static_assert(!totally_ordered<int (S::*)() const volatile&&>, "");
static_assert(!totally_ordered < int (S::*)() const volatile&& noexcept >, "");

static_assert(!totally_ordered<void>, "");
} // namespace fundamentals

namespace standard_types {
static_assert(models_totally_ordered<hip::std::array<int, 10> >(), "");
} // namespace standard_types

namespace types_fit_for_purpose {
#if TEST_STD_VER > 17
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
static_assert(models_totally_ordered<member_three_way_comparable>(), "");
#ifndef __NVCC__  // nvbug3908399
static_assert(models_totally_ordered<friend_three_way_comparable>(), "");
#endif // !__NVCC__
#endif // TEST_HAS_NO_SPACESHIP_OPERATOR

static_assert(models_totally_ordered<explicit_operators>(), "");
static_assert(models_totally_ordered<different_return_types>(), "");
static_assert(!totally_ordered<cxx20_member_eq>, "");
static_assert(!totally_ordered<cxx20_friend_eq>, "");
static_assert(!totally_ordered<one_member_one_friend>, "");
static_assert(!totally_ordered<equality_comparable_with_ec1>, "");
#endif // TEST_STD_VER > 17

static_assert(!totally_ordered<no_eq>, "");
static_assert(!totally_ordered<no_neq>, "");
static_assert(!totally_ordered<no_lt>, "");
static_assert(!totally_ordered<no_gt>, "");
static_assert(!totally_ordered<no_le>, "");
static_assert(!totally_ordered<no_ge>, "");

static_assert(!totally_ordered<wrong_return_type_eq>, "");
static_assert(!totally_ordered<wrong_return_type_ne>, "");
static_assert(!totally_ordered<wrong_return_type_lt>, "");
static_assert(!totally_ordered<wrong_return_type_gt>, "");
static_assert(!totally_ordered<wrong_return_type_le>, "");
static_assert(!totally_ordered<wrong_return_type_ge>, "");
static_assert(!totally_ordered<wrong_return_type>, "");

#if TEST_STD_VER > 17
static_assert(!totally_ordered<cxx20_member_eq_operator_with_deleted_ne>, "");
static_assert(!totally_ordered<cxx20_friend_eq_operator_with_deleted_ne>, "");
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
static_assert(
    !totally_ordered<member_three_way_comparable_with_deleted_eq>, "");
static_assert(
    !totally_ordered<member_three_way_comparable_with_deleted_ne>, "");
static_assert(
    !totally_ordered<friend_three_way_comparable_with_deleted_eq>, "");
#ifndef __NVCC__  // nvbug3908399
static_assert(
    !totally_ordered<friend_three_way_comparable_with_deleted_ne>, "");
#endif // !__NVCC__

static_assert(!totally_ordered<eq_returns_explicit_bool>, "");
static_assert(!totally_ordered<ne_returns_explicit_bool>, "");
static_assert(!totally_ordered<lt_returns_explicit_bool>, "");
static_assert(!totally_ordered<gt_returns_explicit_bool>, "");
static_assert(!totally_ordered<le_returns_explicit_bool>, "");
static_assert(!totally_ordered<ge_returns_explicit_bool>, "");
static_assert(totally_ordered<returns_true_type>, "");
static_assert(totally_ordered<returns_int_ptr>, "");

static_assert(totally_ordered<partial_ordering_totally_ordered_with>, "");
static_assert(totally_ordered<weak_ordering_totally_ordered_with>, "");
static_assert(totally_ordered<strong_ordering_totally_ordered_with>, "");
#endif // TEST_HAS_NO_SPACESHIP_OPERATOR
#endif // TEST_STD_VER > 17
} // namespace types_fit_for_purpose

int main(int, char**) { return 0; }
