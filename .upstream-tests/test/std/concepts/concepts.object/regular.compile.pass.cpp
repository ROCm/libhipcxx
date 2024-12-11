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
// concept regular = see below;
#pragma nv_diag_suppress 3013 // a volatile function parameter is deprecated

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include <hip/std/concepts>

#include "type_classification/moveconstructible.h"
#include "type_classification/semiregular.h"

using hip::std::regular;

static_assert(regular<int>, "");
static_assert(regular<float>, "");
static_assert(regular<double>, "");
static_assert(regular<long double>, "");
static_assert(regular<int volatile>, "");
static_assert(regular<void*>, "");
static_assert(regular<int*>, "");
static_assert(regular<int const*>, "");
static_assert(regular<int volatile*>, "");
static_assert(regular<int volatile const*>, "");
static_assert(regular<int (*)()>, "");

struct S {};
static_assert(!regular<S>, "");
static_assert(regular<int S::*>, "");
static_assert(regular<int (S::*)()>, "");
static_assert(regular<int (S::*)() noexcept>, "");
static_assert(regular<int (S::*)() &>, "");
static_assert(regular<int (S::*)() & noexcept>, "");
static_assert(regular<int (S::*)() &&>, "");
static_assert(regular<int (S::*)() && noexcept>, "");
static_assert(regular<int (S::*)() const>, "");
static_assert(regular<int (S::*)() const noexcept>, "");
static_assert(regular<int (S::*)() const&>, "");
static_assert(regular<int (S::*)() const & noexcept>, "");
static_assert(regular<int (S::*)() const&&>, "");
static_assert(regular<int (S::*)() const && noexcept>, "");
static_assert(regular<int (S::*)() volatile>, "");
static_assert(regular<int (S::*)() volatile noexcept>, "");
static_assert(regular<int (S::*)() volatile&>, "");
static_assert(regular<int (S::*)() volatile & noexcept>, "");
static_assert(regular<int (S::*)() volatile&&>, "");
static_assert(regular<int (S::*)() volatile && noexcept>, "");
static_assert(regular<int (S::*)() const volatile>, "");
static_assert(regular<int (S::*)() const volatile noexcept>, "");
static_assert(regular<int (S::*)() const volatile&>, "");
static_assert(regular<int (S::*)() const volatile & noexcept>, "");
static_assert(regular<int (S::*)() const volatile&&>, "");
static_assert(regular<int (S::*)() const volatile && noexcept>, "");

union U {};
static_assert(!regular<U>, "");
static_assert(regular<int U::*>, "");
static_assert(regular<int (U::*)()>, "");
static_assert(regular<int (U::*)() noexcept>, "");
static_assert(regular<int (U::*)() &>, "");
static_assert(regular<int (U::*)() & noexcept>, "");
static_assert(regular<int (U::*)() &&>, "");
static_assert(regular<int (U::*)() && noexcept>, "");
static_assert(regular<int (U::*)() const>, "");
static_assert(regular<int (U::*)() const noexcept>, "");
static_assert(regular<int (U::*)() const&>, "");
static_assert(regular<int (U::*)() const & noexcept>, "");
static_assert(regular<int (U::*)() const&&>, "");
static_assert(regular<int (U::*)() const && noexcept>, "");
static_assert(regular<int (U::*)() volatile>, "");
static_assert(regular<int (U::*)() volatile noexcept>, "");
static_assert(regular<int (U::*)() volatile&>, "");
static_assert(regular<int (U::*)() volatile & noexcept>, "");
static_assert(regular<int (U::*)() volatile&&>, "");
static_assert(regular<int (U::*)() volatile && noexcept>, "");
static_assert(regular<int (U::*)() const volatile>, "");
static_assert(regular<int (U::*)() const volatile noexcept>, "");
static_assert(regular<int (U::*)() const volatile&>, "");
static_assert(regular<int (U::*)() const volatile & noexcept>, "");
static_assert(regular<int (U::*)() const volatile&&>, "");
static_assert(regular<int (U::*)() const volatile && noexcept>, "");

static_assert(!regular<has_volatile_member>, "");
static_assert(!regular<has_array_member>, "");

// Not objects
static_assert(!regular<void>, "");
static_assert(!regular<int&>, "");
static_assert(!regular<int const&>, "");
static_assert(!regular<int volatile&>, "");
static_assert(!regular<int const volatile&>, "");
static_assert(!regular<int&&>, "");
static_assert(!regular<int const&&>, "");
static_assert(!regular<int volatile&&>, "");
static_assert(!regular<int const volatile&&>, "");
static_assert(!regular<int()>, "");
static_assert(!regular<int (&)()>, "");
static_assert(!regular<int[5]>, "");

// not copyable
static_assert(!regular<int const>, "");
static_assert(!regular<int const volatile>, "");
static_assert(!regular<volatile_copy_assignment volatile>, "");
static_assert(!regular<no_copy_constructor>, "");
static_assert(!regular<no_copy_assignment>, "");
static_assert(!regular<no_copy_assignment_mutable>, "");
static_assert(!regular<derived_from_noncopyable>, "");
static_assert(!regular<has_noncopyable>, "");
static_assert(!regular<has_const_member>, "");
static_assert(!regular<has_cv_member>, "");
static_assert(!regular<has_lvalue_reference_member>, "");
static_assert(!regular<has_rvalue_reference_member>, "");
static_assert(!regular<has_function_ref_member>, "");
static_assert(!regular<deleted_assignment_from_const_rvalue>, "");

// not default_initializable
static_assert(!regular<no_copy_constructor>, "");
static_assert(!regular<no_copy_assignment>, "");
static_assert(hip::std::is_copy_assignable_v<no_copy_assignment_mutable> &&
              !regular<no_copy_assignment_mutable>, "");
static_assert(!regular<derived_from_noncopyable>, "");
static_assert(!regular<has_noncopyable>, "");

static_assert(!regular<derived_from_non_default_initializable>, "");
static_assert(!regular<has_non_default_initializable>, "");

// not equality_comparable
static_assert(!regular<const_copy_assignment const>, "");
static_assert(!regular<cv_copy_assignment const volatile>, "");

struct is_equality_comparable {
  __host__ __device__ bool operator==(is_equality_comparable const&) const { return true; }
  __host__ __device__ bool operator!=(is_equality_comparable const&) const { return false; }
};
static_assert(regular<is_equality_comparable>, "");

int main(int, char**) { return 0; }
