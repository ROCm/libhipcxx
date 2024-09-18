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
// concept movable = see below;
#pragma nv_diag_suppress 3013 // a volatile function parameter is deprecated

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include <hip/std/concepts>

#include "test_macros.h"

#include "type_classification/moveconstructible.h"
#include "type_classification/movable.h"

using hip::std::movable;

// Movable types
static_assert(movable<int>, "");
static_assert(movable<int volatile>, "");
static_assert(movable<int*>, "");
static_assert(movable<int const*>, "");
static_assert(movable<int volatile*>, "");
static_assert(movable<int const volatile*>, "");
static_assert(movable<int (*)()>, "");

struct S {};
static_assert(movable<S>, "");
static_assert(movable<int S::*>, "");
static_assert(movable<int (S::*)()>, "");
static_assert(movable<int (S::*)() noexcept>, "");
static_assert(movable<int (S::*)() &>, "");
static_assert(movable<int (S::*)() & noexcept>, "");
static_assert(movable<int (S::*)() &&>, "");
static_assert(movable<int (S::*)() && noexcept>, "");
static_assert(movable<int (S::*)() const>, "");
static_assert(movable<int (S::*)() const noexcept>, "");
static_assert(movable<int (S::*)() const&>, "");
static_assert(movable<int (S::*)() const & noexcept>, "");
static_assert(movable<int (S::*)() const&&>, "");
static_assert(movable<int (S::*)() const && noexcept>, "");
static_assert(movable<int (S::*)() volatile>, "");
static_assert(movable<int (S::*)() volatile noexcept>, "");
static_assert(movable<int (S::*)() volatile&>, "");
static_assert(movable<int (S::*)() volatile & noexcept>, "");
static_assert(movable<int (S::*)() volatile&&>, "");
static_assert(movable<int (S::*)() volatile && noexcept>, "");
static_assert(movable<int (S::*)() const volatile>, "");
static_assert(movable<int (S::*)() const volatile noexcept>, "");
static_assert(movable<int (S::*)() const volatile&>, "");
static_assert(movable<int (S::*)() const volatile & noexcept>, "");
static_assert(movable<int (S::*)() const volatile&&>, "");
static_assert(movable<int (S::*)() const volatile && noexcept>, "");

static_assert(movable<has_volatile_member>, "");
static_assert(movable<has_array_member>, "");

// Not objects
static_assert(!movable<int&>, "");
static_assert(!movable<int const&>, "");
static_assert(!movable<int volatile&>, "");
static_assert(!movable<int const volatile&>, "");
static_assert(!movable<int&&>, "");
static_assert(!movable<int const&&>, "");
static_assert(!movable<int volatile&&>, "");
static_assert(!movable<int const volatile&&>, "");
static_assert(!movable<int()>, "");
static_assert(!movable<int (&)()>, "");
static_assert(!movable<int[5]>, "");

// Core non-move assignable.
static_assert(!movable<int const>, "");
static_assert(!movable<int const volatile>, "");

static_assert(!movable<DeletedMoveCtor>, "");
static_assert(!movable<ImplicitlyDeletedMoveCtor>, "");
static_assert(!movable<DeletedMoveAssign>, "");
static_assert(!movable<ImplicitlyDeletedMoveAssign>, "");
static_assert(!movable<NonMovable>, "");
static_assert(!movable<DerivedFromNonMovable>, "");
static_assert(!movable<HasANonMovable>, "");

static_assert(movable<cpp03_friendly>, "");
static_assert(movable<const_move_ctor>, "");
static_assert(movable<volatile_move_ctor>, "");
static_assert(movable<cv_move_ctor>, "");
static_assert(movable<multi_param_move_ctor>, "");
static_assert(!movable<not_quite_multi_param_move_ctor>, "");

static_assert(!hip::std::assignable_from<copy_with_mutable_parameter&,
                                    copy_with_mutable_parameter>, "");
static_assert(!movable<copy_with_mutable_parameter>, "");

static_assert(!movable<const_move_assignment>, "");
static_assert(movable<volatile_move_assignment>, "");
static_assert(!movable<cv_move_assignment>, "");

static_assert(!movable<const_move_assign_and_traditional_move_assign>, "");
static_assert(!movable<volatile_move_assign_and_traditional_move_assign>, "");
static_assert(!movable<cv_move_assign_and_traditional_move_assign>, "");
static_assert(movable<const_move_assign_and_default_ops>, "");
static_assert(movable<volatile_move_assign_and_default_ops>, "");
static_assert(movable<cv_move_assign_and_default_ops>, "");

static_assert(!movable<has_const_member>, "");
static_assert(!movable<has_cv_member>, "");
static_assert(!movable<has_lvalue_reference_member>, "");
static_assert(!movable<has_rvalue_reference_member>, "");
static_assert(!movable<has_function_ref_member>, "");

static_assert(movable<deleted_assignment_from_const_rvalue>, "");

// `move_constructible and assignable_from<T&, T>` implies `swappable<T>`,
// so there's nothing to test for the case of non-swappable.

int main(int, char**) { return 0; }
