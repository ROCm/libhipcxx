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
// concept copyable = see below;
#pragma nv_diag_suppress 3013 // a volatile function parameter is deprecated

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include <hip/std/concepts>

#include "type_classification/copyable.h"

using hip::std::copyable;

static_assert(copyable<int>, "");
static_assert(copyable<int volatile>, "");
static_assert(copyable<int*>, "");
static_assert(copyable<int const*>, "");
static_assert(copyable<int volatile*>, "");
static_assert(copyable<int volatile const*>, "");
static_assert(copyable<int (*)()>, "");

struct S {};
static_assert(copyable<S>, "");
static_assert(copyable<int S::*>, "");
static_assert(copyable<int (S::*)()>, "");
static_assert(copyable<int (S::*)() noexcept>, "");
static_assert(copyable<int (S::*)() &>, "");
static_assert(copyable<int (S::*)() & noexcept>, "");
static_assert(copyable<int (S::*)() &&>, "");
static_assert(copyable<int (S::*)() && noexcept>, "");
static_assert(copyable<int (S::*)() const>, "");
static_assert(copyable<int (S::*)() const noexcept>, "");
static_assert(copyable<int (S::*)() const&>, "");
static_assert(copyable<int (S::*)() const & noexcept>, "");
static_assert(copyable<int (S::*)() const&&>, "");
static_assert(copyable<int (S::*)() const && noexcept>, "");
static_assert(copyable<int (S::*)() volatile>, "");
static_assert(copyable<int (S::*)() volatile noexcept>, "");
static_assert(copyable<int (S::*)() volatile&>, "");
static_assert(copyable<int (S::*)() volatile & noexcept>, "");
static_assert(copyable<int (S::*)() volatile&&>, "");
static_assert(copyable<int (S::*)() volatile && noexcept>, "");
static_assert(copyable<int (S::*)() const volatile>, "");
static_assert(copyable<int (S::*)() const volatile noexcept>, "");
static_assert(copyable<int (S::*)() const volatile&>, "");
static_assert(copyable<int (S::*)() const volatile & noexcept>, "");
static_assert(copyable<int (S::*)() const volatile&&>, "");
static_assert(copyable<int (S::*)() const volatile && noexcept>, "");

static_assert(copyable<has_volatile_member>, "");
static_assert(copyable<has_array_member>, "");

// Not objects
static_assert(!copyable<void>, "");
static_assert(!copyable<int&>, "");
static_assert(!copyable<int const&>, "");
static_assert(!copyable<int volatile&>, "");
static_assert(!copyable<int const volatile&>, "");
static_assert(!copyable<int&&>, "");
static_assert(!copyable<int const&&>, "");
static_assert(!copyable<int volatile&&>, "");
static_assert(!copyable<int const volatile&&>, "");
static_assert(!copyable<int()>, "");
static_assert(!copyable<int (&)()>, "");
static_assert(!copyable<int[5]>, "");

// Not assignable
static_assert(!copyable<int const>, "");
static_assert(!copyable<int const volatile>, "");
static_assert(copyable<const_copy_assignment const>, "");
static_assert(!copyable<volatile_copy_assignment volatile>, "");
static_assert(copyable<cv_copy_assignment const volatile>, "");

static_assert(!copyable<no_copy_constructor>, "");
static_assert(!copyable<no_copy_assignment>, "");

static_assert(hip::std::is_copy_assignable_v<no_copy_assignment_mutable>, "");
static_assert(!copyable<no_copy_assignment_mutable>, "");
static_assert(!copyable<derived_from_noncopyable>, "");
static_assert(!copyable<has_noncopyable>, "");
static_assert(!copyable<has_const_member>, "");
static_assert(!copyable<has_cv_member>, "");
static_assert(!copyable<has_lvalue_reference_member>, "");
static_assert(!copyable<has_rvalue_reference_member>, "");
static_assert(!copyable<has_function_ref_member>, "");

static_assert(
    !hip::std::assignable_from<deleted_assignment_from_const_rvalue&,
                          deleted_assignment_from_const_rvalue const>, "");
static_assert(!copyable<deleted_assignment_from_const_rvalue>, "");

int main(int, char**) { return 0; }
