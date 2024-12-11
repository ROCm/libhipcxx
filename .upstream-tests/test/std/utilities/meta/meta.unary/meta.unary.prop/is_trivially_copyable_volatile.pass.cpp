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

// type_traits

// is_trivially_copyable for volatile qualified types

// These compilers have not implemented Core 2094 which makes volatile
// qualified types trivially copyable.
// XFAIL: clang-3, clang-4, apple-clang-6, apple-clang-7, apple-clang-8, apple-clang-9.0
// XFAIL: gcc-4.8, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9
// XFAIL: icc
// XFAIL: pgi

// When we marked this XFAIL for MSVC, QA reported that it unexpectedly passed.
// When we stopped marking it XFAIL for MSVC, QA reported that it unexpectedly
// failed
// UNSUPPORTED: msvc

// Note that this passes under NVRTC, so don't mark this as expected failure
// for NVCC, because that'll get us the wrong result under runtime compilation
// (since NVRTC also has the `nvcc` feature). (It's possible that the compiler
// detection should be changed to not report `nvcc` for NVRTC, but that's a
// topic for another day.)

// Because nvrtc inherits flags from the host compilers, gcc-7 expects failure while NVRTC builds pass.
// Mark unsupported because the matrix of all {compiler && !nvrtc} is insane
// UNSUPPORTED: nvrtc

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

#include <hip/std/type_traits>
#include <hip/std/cassert>
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_trivially_copyable_volatile()
{
    static_assert( hip::std::is_trivially_copyable<volatile T>::value, "");
    static_assert( hip::std::is_trivially_copyable<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_trivially_copyable_v<volatile T>, "");
    static_assert( hip::std::is_trivially_copyable_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_trivially_copyable_volatile()
{
    static_assert(!hip::std::is_trivially_copyable<volatile T>::value, "");
    static_assert(!hip::std::is_trivially_copyable<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_trivially_copyable_v<volatile T>, "");
    static_assert(!hip::std::is_trivially_copyable_v<const volatile T>, "");
#endif
}

struct A
{
    int i_;
};

struct B
{
    int i_;
    __host__ __device__
    ~B() {assert(i_ == 0);}
};

class C
{
public:
    __host__ __device__
    C();
};

int main(int, char**)
{
    test_is_trivially_copyable_volatile<int> ();
    test_is_trivially_copyable_volatile<const int> ();
    test_is_trivially_copyable_volatile<A> ();
    test_is_trivially_copyable_volatile<const A> ();
    test_is_trivially_copyable_volatile<C> ();

    test_is_not_trivially_copyable_volatile<int&> ();
    test_is_not_trivially_copyable_volatile<const A&> ();
    test_is_not_trivially_copyable_volatile<B> ();

    return 0;
}
