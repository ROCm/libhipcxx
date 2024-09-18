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

// is_nothrow_destructible

// Prevent warning when testing the Abstract test type.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdelete-non-virtual-dtor"
#endif

#include <hip/std/type_traits>
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_nothrow_destructible()
{
    static_assert( hip::std::is_nothrow_destructible<T>::value, "");
    static_assert( hip::std::is_nothrow_destructible<const T>::value, "");
    static_assert( hip::std::is_nothrow_destructible<volatile T>::value, "");
    static_assert( hip::std::is_nothrow_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_nothrow_destructible_v<T>, "");
    static_assert( hip::std::is_nothrow_destructible_v<const T>, "");
    static_assert( hip::std::is_nothrow_destructible_v<volatile T>, "");
    static_assert( hip::std::is_nothrow_destructible_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_nothrow_destructible()
{
    static_assert(!hip::std::is_nothrow_destructible<T>::value, "");
    static_assert(!hip::std::is_nothrow_destructible<const T>::value, "");
    static_assert(!hip::std::is_nothrow_destructible<volatile T>::value, "");
    static_assert(!hip::std::is_nothrow_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_nothrow_destructible_v<T>, "");
    static_assert(!hip::std::is_nothrow_destructible_v<const T>, "");
    static_assert(!hip::std::is_nothrow_destructible_v<volatile T>, "");
    static_assert(!hip::std::is_nothrow_destructible_v<const volatile T>, "");
#endif
}


struct PublicDestructor           { public:     __host__ __device__ ~PublicDestructor() {}};
struct ProtectedDestructor        { protected:  __host__ __device__ ~ProtectedDestructor() {}};
struct PrivateDestructor          { private:    __host__ __device__ ~PrivateDestructor() {}};

struct VirtualPublicDestructor           { public:    __host__ __device__ virtual ~VirtualPublicDestructor() {}};
struct VirtualProtectedDestructor        { protected: __host__ __device__ virtual ~VirtualProtectedDestructor() {}};
struct VirtualPrivateDestructor          { private:   __host__ __device__ virtual ~VirtualPrivateDestructor() {}};

struct PurePublicDestructor              { public:    __host__ __device__ virtual ~PurePublicDestructor() = 0; };
struct PureProtectedDestructor           { protected: __host__ __device__ virtual ~PureProtectedDestructor() = 0; };
struct PurePrivateDestructor             { private:   __host__ __device__ virtual ~PurePrivateDestructor() = 0; };

class Empty
{
};


union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    __host__ __device__
    virtual void foo() = 0;
};


int main(int, char**)
{
    test_is_not_nothrow_destructible<void>();
    test_is_not_nothrow_destructible<char[]>();
    test_is_not_nothrow_destructible<char[][3]>();

    test_is_nothrow_destructible<int&>();
    test_is_nothrow_destructible<int>();
    test_is_nothrow_destructible<double>();
    test_is_nothrow_destructible<int*>();
    test_is_nothrow_destructible<const int*>();
    test_is_nothrow_destructible<char[3]>();

#if TEST_STD_VER >= 11
    // requires noexcept. These are all destructible.
    test_is_nothrow_destructible<PublicDestructor>();
    test_is_nothrow_destructible<VirtualPublicDestructor>();
    test_is_nothrow_destructible<PurePublicDestructor>();
    test_is_nothrow_destructible<bit_zero>();
    test_is_nothrow_destructible<Abstract>();
    test_is_nothrow_destructible<Empty>();
    test_is_nothrow_destructible<Union>();
#endif
    // requires access control
    test_is_not_nothrow_destructible<ProtectedDestructor>();
    test_is_not_nothrow_destructible<PrivateDestructor>();
    test_is_not_nothrow_destructible<VirtualProtectedDestructor>();
    test_is_not_nothrow_destructible<VirtualPrivateDestructor>();
    test_is_not_nothrow_destructible<PureProtectedDestructor>();
    test_is_not_nothrow_destructible<PurePrivateDestructor>();


  return 0;
}
