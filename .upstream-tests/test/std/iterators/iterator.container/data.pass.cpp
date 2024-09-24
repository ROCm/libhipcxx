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

// UNSUPPORTED: c++98, c++03, c++11

// <cuda/std/iterator>
// template <class C> constexpr auto data(C& c) -> decltype(c.data());               // C++17
// template <class C> constexpr auto data(const C& c) -> decltype(c.data());         // C++17
// template <class T, size_t N> constexpr T* data(T (&array)[N]) noexcept;           // C++17
// template <class E> constexpr const E* data(initializer_list<E> il) noexcept;      // C++17

#include <hip/std/iterator>
#include <hip/std/cassert>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#include <hip/std/vector>
#endif
#include <hip/std/array>
#include <hip/std/initializer_list>

#include "test_macros.h"

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#if TEST_STD_VER > 14
#include <hip/std/string_view>
#endif
#endif

template<typename C>
__host__ __device__
void test_const_container( const C& c )
{
//  Can't say noexcept here because the container might not be
    assert ( hip::std::data(c)   == c.data());
}

template<typename T>
__host__ __device__
void test_const_container( const hip::std::initializer_list<T>& c )
{
    ASSERT_NOEXCEPT(hip::std::data(c));
    assert ( hip::std::data(c)   == c.begin());
}

template<typename C>
__host__ __device__
void test_container( C& c )
{
//  Can't say noexcept here because the container might not be
    assert ( hip::std::data(c)   == c.data());
}

template<typename T>
__host__ __device__
void test_container( hip::std::initializer_list<T>& c)
{
    ASSERT_NOEXCEPT(hip::std::data(c));
    assert ( hip::std::data(c)   == c.begin());
}

template<typename T, size_t Sz>
__host__ __device__
void test_const_array( const T (&array)[Sz] )
{
    ASSERT_NOEXCEPT(hip::std::data(array));
    assert ( hip::std::data(array) == &array[0]);
}

__device__ static constexpr int arrA [] { 1, 2, 3 };

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_VECTOR)
    hip::std::vector<int> v; v.push_back(1);
#endif
    hip::std::array<int, 1> a; a[0] = 3;
    hip::std::initializer_list<int> il = { 4 };

#if defined(_LIBCUDACXX_HAS_VECTOR)
    test_container ( v );
#endif
    test_container ( a );
    test_container ( il );

#if defined(_LIBCUDACXX_HAS_VECTOR)
    test_const_container ( v );
#endif
    test_const_container ( a );
    test_const_container ( il );

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#if TEST_STD_VER > 14
    hip::std::string_view sv{"ABC"};
    test_container ( sv );
    test_const_container ( sv );
#endif
#endif

    test_const_array ( arrA );

  return 0;
}
