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

#include <hip/std/cstddef>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

// typedef decltype(nullptr) nullptr_t;

struct A
{
    __host__ __device__
    A(hip::std::nullptr_t) {}
};

template <class T>
__host__ __device__
void test_conversions()
{
    {
        T p = 0;
        assert(p == nullptr);
        (void)p; // GCC spuriously claims that p is unused when T is nullptr_t, probably due to optimizations?
    }
    {
        T p = nullptr;
        assert(p == nullptr);
        assert(nullptr == p);
        assert(!(p != nullptr));
        assert(!(nullptr != p));
        (void)p; // GCC spuriously claims that p is unused when T is nullptr_t, probably due to optimizations?
    }
}

template <class T> struct Voider { typedef void type; };
template <class T, class = void> struct has_less : hip::std::false_type {};

template <class T> struct has_less<T,
    typename Voider<decltype(hip::std::declval<T>() < nullptr)>::type> : hip::std::true_type {};

template <class T>
__host__ __device__
void test_comparisons()
{
    T p = nullptr;
    assert(p == nullptr);
    assert(!(p != nullptr));
    assert(nullptr == p);
    assert(!(nullptr != p));
    (void)p; // GCC spuriously claims that p is unused, probably due to optimizations?
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnull-conversion"
#endif
__host__ __device__
void test_nullptr_conversions() {
// GCC does not accept this due to CWG Defect #1423
// http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1423
#if defined(__clang__) && !defined(TEST_COMPILER_NVCC) && !defined(TEST_COMPILER_HIPCC)
    {
        bool b = nullptr;
        assert(!b);
    }
#endif
    {
        bool b(nullptr);
        assert(!b);
    }
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif


int main(int, char**)
{
    static_assert(sizeof(hip::std::nullptr_t) == sizeof(void*),
                  "sizeof(hip::std::nullptr_t) == sizeof(void*)");

    {
        test_conversions<hip::std::nullptr_t>();
        test_conversions<void*>();
        test_conversions<A*>();
        test_conversions<void(*)()>();
        test_conversions<void(A::*)()>();
        test_conversions<int A::*>();
    }
    {
#ifdef _LIBCUDACXX_HAS_NO_NULLPTR
        static_assert(!has_less<hip::std::nullptr_t>::value, "");
        // FIXME: our C++03 nullptr emulation still allows for comparisons
        // with other pointer types by way of the conversion operator.
        //static_assert(!has_less<void*>::value, "");
#else
        // TODO Enable this assertion when all compilers implement core DR 583.
        // static_assert(!has_less<hip::std::nullptr_t>::value, "");
#endif
        test_comparisons<hip::std::nullptr_t>();
        test_comparisons<void*>();
        test_comparisons<A*>();
        test_comparisons<void(*)()>();
    }
    test_nullptr_conversions();

  return 0;
}
