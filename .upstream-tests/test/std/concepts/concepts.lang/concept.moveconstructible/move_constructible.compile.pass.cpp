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
// concept move_constructible;
#pragma nv_diag_suppress 3013 // a volatile function parameter is deprecated

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include <hip/std/concepts>
#include <hip/std/type_traits>

#include "type_classification/moveconstructible.h"

using hip::std::move_constructible;

static_assert(move_constructible<int>, "");
static_assert(move_constructible<int*>, "");
static_assert(move_constructible<int&>, "");
static_assert(move_constructible<int&&>, "");
static_assert(move_constructible<const int>, "");
static_assert(move_constructible<const int&>, "");
static_assert(move_constructible<const int&&>, "");
static_assert(move_constructible<volatile int>, "");
static_assert(move_constructible<volatile int&>, "");
static_assert(move_constructible<volatile int&&>, "");
static_assert(move_constructible<int (*)()>, "");
static_assert(move_constructible<int (&)()>, "");
static_assert(move_constructible<HasDefaultOps>, "");
static_assert(move_constructible<CustomMoveCtor>, "");
static_assert(move_constructible<MoveOnly>, "");
static_assert(move_constructible<const CustomMoveCtor&>, "");
static_assert(move_constructible<volatile CustomMoveCtor&>, "");
static_assert(move_constructible<const CustomMoveCtor&&>, "");
static_assert(move_constructible<volatile CustomMoveCtor&&>, "");
static_assert(move_constructible<CustomMoveAssign>, "");
static_assert(move_constructible<const CustomMoveAssign&>, "");
static_assert(move_constructible<volatile CustomMoveAssign&>, "");
static_assert(move_constructible<const CustomMoveAssign&&>, "");
static_assert(move_constructible<volatile CustomMoveAssign&&>, "");
static_assert(move_constructible<int HasDefaultOps::*>, "");
static_assert(move_constructible<void (HasDefaultOps::*)(int)>, "");
static_assert(move_constructible<MemberLvalueReference>, "");
static_assert(move_constructible<MemberRvalueReference>, "");

static_assert(!move_constructible<void>, "");
static_assert(!move_constructible<const CustomMoveCtor>, "");
static_assert(!move_constructible<volatile CustomMoveCtor>, "");
static_assert(!move_constructible<const CustomMoveAssign>, "");
static_assert(!move_constructible<volatile CustomMoveAssign>, "");
static_assert(!move_constructible<int[10]>, "");
static_assert(!move_constructible<DeletedMoveCtor>, "");
static_assert(!move_constructible<ImplicitlyDeletedMoveCtor>, "");
static_assert(!move_constructible<DeletedMoveAssign>, "");
static_assert(!move_constructible<ImplicitlyDeletedMoveAssign>, "");

static_assert(move_constructible<DeletedMoveCtor&>, "");
static_assert(move_constructible<DeletedMoveCtor&&>, "");
static_assert(move_constructible<const DeletedMoveCtor&>, "");
static_assert(move_constructible<const DeletedMoveCtor&&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveCtor&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveCtor&&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveCtor&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveCtor&&>, "");
static_assert(move_constructible<DeletedMoveAssign&>, "");
static_assert(move_constructible<DeletedMoveAssign&&>, "");
static_assert(move_constructible<const DeletedMoveAssign&>, "");
static_assert(move_constructible<const DeletedMoveAssign&&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveAssign&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveAssign&&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveAssign&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveAssign&&>, "");

static_assert(!move_constructible<NonMovable>, "");
static_assert(!move_constructible<DerivedFromNonMovable>, "");
static_assert(!move_constructible<HasANonMovable>, "");

int main(int, char**) { return 0; }
