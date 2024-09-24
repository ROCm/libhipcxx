//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
//

// __libcpp_is_referenceable<Tp>
//
// [defns.referenceable] defines "a referenceable type" as:
// An object type, a function type that does not have cv-qualifiers
//    or a ref-qualifier, or a reference type.
//

#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

struct Foo {};

static_assert((!hip::std::__libcpp_is_referenceable<void>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<int>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<int[3]>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<int[]>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<int &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<const int &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<int *>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<const int *>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<Foo>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<const Foo>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<Foo &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<const Foo &>::value), "");
#if TEST_STD_VER >= 11
static_assert(( hip::std::__libcpp_is_referenceable<Foo &&>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<const Foo &&>::value), "");
#endif

#ifndef  _LIBCUDACXX_HAS_NO_VECTOR_EXTENSION
static_assert(( hip::std::__libcpp_is_referenceable<int   __attribute__((__vector_size__( 8)))>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<const int   __attribute__((__vector_size__( 8)))>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<float __attribute__((__vector_size__(16)))>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<const float __attribute__((__vector_size__(16)))>::value), "");
#endif

// Functions without cv-qualifiers are referenceable
static_assert(( hip::std::__libcpp_is_referenceable<void ()>::value), "");
#if TEST_STD_VER >= 11
static_assert((!hip::std::__libcpp_is_referenceable<void () const>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void () &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void () const &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void () &&>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void () const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (int)>::value), "");
#if TEST_STD_VER >= 11
static_assert((!hip::std::__libcpp_is_referenceable<void (int) const>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int) &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int) const &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int) &&>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (int, float)>::value), "");
#if TEST_STD_VER >= 11
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float) const>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float) &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float) const &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float) &&>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (int, float, Foo &)>::value), "");
#if TEST_STD_VER >= 11
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &) const>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &) &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &) const &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &) &&>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (...)>::value), "");
#if TEST_STD_VER >= 11
static_assert((!hip::std::__libcpp_is_referenceable<void (...) const>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (...) &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (...) const &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (...) &&>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (...) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (int, ...)>::value), "");
#if TEST_STD_VER >= 11
static_assert((!hip::std::__libcpp_is_referenceable<void (int, ...) const>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, ...) &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, ...) const &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, ...) &&>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, ...) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (int, float, ...)>::value), "");
#if TEST_STD_VER >= 11
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, ...) const>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, ...) &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, ...) const &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, ...) &&>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, ...) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (int, float, Foo &, ...)>::value), "");
#if TEST_STD_VER >= 11
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &, ...) const>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &, ...) &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &, ...) const &>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &, ...) &&>::value), "");
static_assert((!hip::std::__libcpp_is_referenceable<void (int, float, Foo &, ...) const &&>::value), "");
#endif

// member functions with or without cv-qualifiers are referenceable
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)()>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)() const>::value), "");
#if TEST_STD_VER >= 11
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)() &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)() const &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)() &&>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)() const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int)>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int) const>::value), "");
#if TEST_STD_VER >= 11
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int) &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int) const &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int) &&>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float)>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float) const>::value), "");
#if TEST_STD_VER >= 11
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float) &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float) const &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float) &&>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &)>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &) const>::value), "");
#if TEST_STD_VER >= 11
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &) &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &) const &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &) &&>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(...)>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(...) const>::value), "");
#if TEST_STD_VER >= 11
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(...) &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(...) const &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(...) &&>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(...) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, ...)>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, ...) const>::value), "");
#if TEST_STD_VER >= 11
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, ...) &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, ...) const &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, ...) &&>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, ...) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, ...)>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, ...) const>::value), "");
#if TEST_STD_VER >= 11
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, ...) &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, ...) const &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, ...) &&>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, ...) const &&>::value), "");
#endif

static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &, ...)>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &, ...) const>::value), "");
#if TEST_STD_VER >= 11
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &, ...) &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &, ...) const &>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &, ...) &&>::value), "");
static_assert(( hip::std::__libcpp_is_referenceable<void (Foo::*)(int, float, Foo &, ...) const &&>::value), "");
#endif

int main(int, char**) {
  return 0;
}
