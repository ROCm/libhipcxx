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

// alignment_of

#include <hip/std/type_traits>
#include <hip/std/cstdint>

#include "test_macros.h"

template <class T, unsigned A>
__host__ __device__
void test_alignment_of()
{
    const unsigned AlignofResult = TEST_ALIGNOF(T);
    static_assert( AlignofResult == A, "Golden value does not match result of alignof keyword");
    static_assert( hip::std::alignment_of<T>::value == AlignofResult, "");
    static_assert( hip::std::alignment_of<T>::value == A, "");
    static_assert( hip::std::alignment_of<const T>::value == A, "");
    static_assert( hip::std::alignment_of<volatile T>::value == A, "");
    static_assert( hip::std::alignment_of<const volatile T>::value == A, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::alignment_of_v<T> == A, "");
    static_assert( hip::std::alignment_of_v<const T> == A, "");
    static_assert( hip::std::alignment_of_v<volatile T> == A, "");
    static_assert( hip::std::alignment_of_v<const volatile T> == A, "");
#endif
}

class Class
{
public:
    __host__ __device__
    ~Class();
};

int main(int, char**)
{
    test_alignment_of<int&, 4>();
    test_alignment_of<Class, 1>();
    test_alignment_of<int*, sizeof(intptr_t)>();
    test_alignment_of<const int*, sizeof(intptr_t)>();
    test_alignment_of<char[3], 1>();
    test_alignment_of<int, 4>();
    // The test case below is a hack. It's hard to detect what golden value
    // we should expect. In most cases it should be 8. But in i386 builds
    // with Clang >= 8 or GCC >= 8 the value is '4'.
    test_alignment_of<double, TEST_ALIGNOF(double)>();
#if (defined(__ppc__) && !defined(__ppc64__))
    test_alignment_of<bool, 4>();   // 32-bit PPC has four byte bool
#else
    test_alignment_of<bool, 1>();
#endif
    test_alignment_of<unsigned, 4>();

  return 0;
}
