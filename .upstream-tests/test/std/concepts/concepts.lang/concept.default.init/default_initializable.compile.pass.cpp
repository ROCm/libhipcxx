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

// We voluntarily use hip::std::default_initializable on types that have redundant
// or ignored cv-qualifiers -- don't warn about it.
// ADDITIONAL_COMPILE_FLAGS: -Wno-ignored-qualifiers

// template<class T>
//     concept default_initializable = constructible_from<T> &&
//     requires { T{}; } &&
//     is-default-initializable<T>;

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include <hip/std/array>
#include <hip/std/concepts>

#include "test_macros.h"

struct Empty {};

struct CtorDefaulted {
  CtorDefaulted() = default;
};
struct CtorDeleted {
  CtorDeleted() = delete;
};
struct DtorDefaulted {
  ~DtorDefaulted() = default;
};
struct DtorDeleted {
  ~DtorDeleted() = delete;
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

struct CtorProtected {
protected:
  CtorProtected() = default;
};
struct CtorPrivate {
private:
  CtorPrivate() = default;
};
struct DtorProtected {
protected:
  ~DtorProtected() = default;
};
struct DtorPrivate {
private:
  ~DtorPrivate() = default;
};

template <class T>
struct NoexceptDependant {
  __host__ __device__ ~NoexceptDependant() noexcept(hip::std::is_same_v<T, int>);
};

struct CtorExplicit {
  explicit CtorExplicit() = default;
};
struct CtorArgument {
  __host__ __device__ CtorArgument(int) {}
};
struct CtorDefaultArgument {
  __host__ __device__ CtorDefaultArgument(int = 0) {}
};
struct CtorExplicitDefaultArgument {
  __host__ __device__ explicit CtorExplicitDefaultArgument(int = 0) {}
};

struct Derived : public Empty {};

class Abstract {
  __host__ __device__ virtual void foo() = 0;
};

class AbstractDestructor {
  __host__ __device__ virtual ~AbstractDestructor() = 0;
};

class OperatorNewDeleted {
  void* operator new(hip::std::size_t) = delete;
  void operator delete(void* ptr) = delete;
};

template<class T>
__host__ __device__ void test_not_const()
{
    static_assert( hip::std::default_initializable<               T>, "");
    static_assert( hip::std::default_initializable<      volatile T>, "");
#if !defined(TEST_COMPILER_C1XX) // nvbug3953465
    static_assert(!hip::std::default_initializable<const          T>, "");
    static_assert(!hip::std::default_initializable<const volatile T>, "");
#endif
}

template<class T>
__host__ __device__ void test_true()
{
    static_assert( hip::std::default_initializable<               T>, "");
    static_assert( hip::std::default_initializable<const          T>, "");
    static_assert( hip::std::default_initializable<      volatile T>, "");
    static_assert( hip::std::default_initializable<const volatile T>, "");
}

template<class T>
__host__ __device__ void test_false()
{
    static_assert(!hip::std::default_initializable<               T>, "");
    static_assert(!hip::std::default_initializable<const          T>, "");
    static_assert(!hip::std::default_initializable<      volatile T>, "");
    static_assert(!hip::std::default_initializable<const volatile T>, "");
}

__host__ __device__ void test()
{
    test_not_const<bool>();
    test_not_const<char>();
    test_not_const<int>();
    test_not_const<double>();

    test_false    <void>();
    test_not_const<void*>();

    test_not_const<int*>();
    test_false    <int[]>();
    test_not_const<int[1]>();
    test_false    <int&>();
    test_false    <int&&>();

    test_true     <Empty>();

    test_true     <CtorDefaulted>();
    test_false    <CtorDeleted>();
    test_true     <DtorDefaulted>();
    test_false    <DtorDeleted>();

    test_true     <Noexcept>();
    test_true     <NoexceptTrue>();
    test_false    <NoexceptFalse>();

    test_false    <CtorProtected>();
    test_false    <CtorPrivate>();
    test_false    <DtorProtected>();
    test_false    <DtorPrivate>();

    test_true     <NoexceptDependant<int>>();
    test_false    <NoexceptDependant<double>>();

    test_true     <CtorExplicit>();
    test_false    <CtorArgument>();
    test_true     <CtorDefaultArgument>();
    test_true     <CtorExplicitDefaultArgument>();

    test_true     <Derived>();
    test_false    <Abstract>();
    test_false    <AbstractDestructor>();

    test_true     <OperatorNewDeleted>();

#if !defined(__GNUC__) || (__GNUC__ > 11) // type qualifiers ignored on cast result type [-Werror=ignored-qualifiers]
    test_not_const<void(*)(const int&)>();
    test_not_const<void(Empty::*)(const int&)               >();
    test_not_const<void(Empty::*)(const int&) const         >();
    test_not_const<void(Empty::*)(const int&)       volatile>();
    test_not_const<void(Empty::*)(const int&) const volatile>();
    test_not_const<void(Empty::*)(const int&) &>();
    test_not_const<void(Empty::*)(const int&) &&>();
    test_not_const<void(Empty::*)(const int&) noexcept>();
    test_not_const<void(Empty::*)(const int&) noexcept(true)>();
    test_not_const<void(Empty::*)(const int&) noexcept(false)>();
#endif

    // Sequence containers
    test_not_const<hip::std::array<               int, 0>>();
    test_not_const<hip::std::array<               int, 1>>();
    test_false    <hip::std::array<const          int, 1>>();
    test_not_const<hip::std::array<      volatile int, 1>>();
    test_false    <hip::std::array<const volatile int, 1>>();
}

// Required for MSVC internal test runner compatibility.
int main(int, char**) {
    return 0;
}
