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

// XFAIL: c++98, c++03

// <cuda/std/iterator>
// template <class C> constexpr auto begin(C& c) -> decltype(c.begin());
// template <class C> constexpr auto begin(const C& c) -> decltype(c.begin());
// template <class C> constexpr auto cbegin(const C& c) -> decltype(hip::std::begin(c)); // C++14
// template <class C> constexpr auto cend(const C& c) -> decltype(hip::std::end(c));     // C++14
// template <class C> constexpr auto end  (C& c) -> decltype(c.end());
// template <class C> constexpr auto end  (const C& c) -> decltype(c.end());
// template <class E> constexpr reverse_iterator<const E*> rbegin(initializer_list<E> il);
// template <class E> constexpr reverse_iterator<const E*> rend  (initializer_list<E> il);
//
// template <class C> auto constexpr rbegin(C& c) -> decltype(c.rbegin());                 // C++14
// template <class C> auto constexpr rbegin(const C& c) -> decltype(c.rbegin());           // C++14
// template <class C> auto constexpr rend(C& c) -> decltype(c.rend());                     // C++14
// template <class C> constexpr auto rend(const C& c) -> decltype(c.rend());               // C++14
// template <class T, size_t N> reverse_iterator<T*> constexpr rbegin(T (&array)[N]);      // C++14
// template <class T, size_t N> reverse_iterator<T*> constexpr rend(T (&array)[N]);        // C++14
// template <class C> constexpr auto crbegin(const C& c) -> decltype(hip::std::rbegin(c));      // C++14
// template <class C> constexpr auto crend(const C& c) -> decltype(hip::std::rend(c));          // C++14
//
//  All of these are constexpr in C++17

#include <hip/std/iterator>
#include <hip/std/cassert>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#include <hip/std/vector>
#endif
#include <hip/std/array>
#if defined(_LIBCUDACXX_HAS_LIST)
#include <hip/std/list>
#endif
#include <hip/std/initializer_list>

#include "test_macros.h"

// hip::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

template<typename C>
__host__ __device__
void test_const_container( const C & c, typename C::value_type val ) {
    assert ( hip::std::begin(c)   == c.begin());
    assert (*hip::std::begin(c)   ==  val );
    assert ( hip::std::begin(c)   != c.end());
    assert ( hip::std::end(c)     == c.end());
#if TEST_STD_VER > 11
    assert ( hip::std::cbegin(c)  == c.cbegin());
    assert ( hip::std::cbegin(c)  != c.cend());
    assert ( hip::std::cend(c)    == c.cend());
    assert ( hip::std::rbegin(c)  == c.rbegin());
    assert ( hip::std::rbegin(c)  != c.rend());
    assert ( hip::std::rend(c)    == c.rend());
    assert ( hip::std::crbegin(c) == c.crbegin());
    assert ( hip::std::crbegin(c) != c.crend());
    assert ( hip::std::crend(c)   == c.crend());
#endif
    }

template<typename T>
__host__ __device__
void test_const_container( const hip::std::initializer_list<T> & c, T val ) {
    assert ( hip::std::begin(c)   == c.begin());
    assert (*hip::std::begin(c)   ==  val );
    assert ( hip::std::begin(c)   != c.end());
    assert ( hip::std::end(c)     == c.end());
#if TEST_STD_VER > 11
//  initializer_list doesn't have cbegin/cend/rbegin/rend
//  but hip::std::cbegin(),etc work (b/c they're general fn templates)
//     assert ( hip::std::cbegin(c)  == c.cbegin());
//     assert ( hip::std::cbegin(c)  != c.cend());
//     assert ( hip::std::cend(c)    == c.cend());
//     assert ( hip::std::rbegin(c)  == c.rbegin());
//     assert ( hip::std::rbegin(c)  != c.rend());
//     assert ( hip::std::rend(c)    == c.rend());
//     assert ( hip::std::crbegin(c) == c.crbegin());
//     assert ( hip::std::crbegin(c) != c.crend());
//     assert ( hip::std::crend(c)   == c.crend());
#endif
    }

template<typename C>
__host__ __device__
void test_container( C & c, typename C::value_type val ) {
    assert ( hip::std::begin(c)   == c.begin());
    assert (*hip::std::begin(c)   ==  val );
    assert ( hip::std::begin(c)   != c.end());
    assert ( hip::std::end(c)     == c.end());
#if TEST_STD_VER > 11
    assert ( hip::std::cbegin(c)  == c.cbegin());
    assert ( hip::std::cbegin(c)  != c.cend());
    assert ( hip::std::cend(c)    == c.cend());
    assert ( hip::std::rbegin(c)  == c.rbegin());
    assert ( hip::std::rbegin(c)  != c.rend());
    assert ( hip::std::rend(c)    == c.rend());
    assert ( hip::std::crbegin(c) == c.crbegin());
    assert ( hip::std::crbegin(c) != c.crend());
    assert ( hip::std::crend(c)   == c.crend());
#endif
    }

template<typename T>
__host__ __device__
void test_container( hip::std::initializer_list<T> & c, T val ) {
    assert ( hip::std::begin(c)   == c.begin());
    assert (*hip::std::begin(c)   ==  val );
    assert ( hip::std::begin(c)   != c.end());
    assert ( hip::std::end(c)     == c.end());
#if TEST_STD_VER > 11
//  initializer_list doesn't have cbegin/cend/rbegin/rend
//     assert ( hip::std::cbegin(c)  == c.cbegin());
//     assert ( hip::std::cbegin(c)  != c.cend());
//     assert ( hip::std::cend(c)    == c.cend());
//     assert ( hip::std::rbegin(c)  == c.rbegin());
//     assert ( hip::std::rbegin(c)  != c.rend());
//     assert ( hip::std::rend(c)    == c.rend());
//     assert ( hip::std::crbegin(c) == c.crbegin());
//     assert ( hip::std::crbegin(c) != c.crend());
//     assert ( hip::std::crend(c)   == c.crend());
#endif
    }

template<typename T, size_t Sz>
__host__ __device__
void test_const_array( const T (&array)[Sz] ) {
    assert ( hip::std::begin(array)  == array );
    assert (*hip::std::begin(array)  ==  array[0] );
    assert ( hip::std::begin(array)  != hip::std::end(array));
    assert ( hip::std::end(array)    == array + Sz);
#if TEST_STD_VER > 11
    assert ( hip::std::cbegin(array) == array );
    assert (*hip::std::cbegin(array) == array[0] );
    assert ( hip::std::cbegin(array) != hip::std::cend(array));
    assert ( hip::std::cend(array)   == array + Sz);
#endif
    }

__device__ static constexpr int arrA [] { 1, 2, 3 };
#if TEST_STD_VER > 14
__device__ static constexpr const int c[] = {0,1,2,3,4};
#endif

int main(int, char**) {
#if defined(_LIBCUDACXX_HAS_VECTOR)
    hip::std::vector<int> v; v.push_back(1);
#endif
#if defined(_LIBCUDACXX_HAS_LIST)
    hip::std::list<int> l;   l.push_back(2);
#endif
    hip::std::array<int, 1> a; a[0] = 3;
    hip::std::initializer_list<int> il = { 4 };

#if defined(_LIBCUDACXX_HAS_VECTOR)
    test_container ( v, 1 );
#endif
#if defined(_LIBCUDACXX_HAS_LIST)
    test_container ( l, 2 );
#endif
    test_container ( a, 3 );
    test_container ( il, 4 );

#if defined(_LIBCUDACXX_HAS_VECTOR)
    test_const_container ( v, 1 );
#endif
#if defined(_LIBCUDACXX_HAS_LIST)
    test_const_container ( l, 2 );
#endif
    test_const_container ( a, 3 );
    test_const_container ( il, 4 );

    test_const_array ( arrA );
#if TEST_STD_VER > 11
    constexpr const int *b = hip::std::cbegin(arrA);
    constexpr const int *e = hip::std::cend(arrA);
    static_assert(e - b == 3, "");
#endif

#if TEST_STD_VER > 14
    {
        typedef hip::std::array<int, 5> C;
        constexpr const C c{0,1,2,3,4};

        static_assert ( c.begin()   == hip::std::begin(c), "");
        static_assert ( c.cbegin()  == hip::std::cbegin(c), "");
        static_assert ( c.end()     == hip::std::end(c), "");
        static_assert ( c.cend()    == hip::std::cend(c), "");

        static_assert ( c.rbegin()  == hip::std::rbegin(c), "");
        static_assert ( c.crbegin() == hip::std::crbegin(c), "");
        static_assert ( c.rend()    == hip::std::rend(c), "");
        static_assert ( c.crend()   == hip::std::crend(c), "");

        static_assert ( hip::std::begin(c)   != hip::std::end(c), "");
        static_assert ( hip::std::rbegin(c)  != hip::std::rend(c), "");
        static_assert ( hip::std::cbegin(c)  != hip::std::cend(c), "");
        static_assert ( hip::std::crbegin(c) != hip::std::crend(c), "");

        static_assert ( *c.begin()  == 0, "");
        static_assert ( *c.rbegin()  == 4, "");

        static_assert ( *hip::std::begin(c)   == 0, "" );
        static_assert ( *hip::std::cbegin(c)  == 0, "" );
        static_assert ( *hip::std::rbegin(c)  == 4, "" );
        static_assert ( *hip::std::crbegin(c) == 4, "" );
    }

    {

        static_assert ( *hip::std::begin(c)   == 0, "" );
        static_assert ( *hip::std::cbegin(c)  == 0, "" );
        static_assert ( *hip::std::rbegin(c)  == 4, "" );
        static_assert ( *hip::std::crbegin(c) == 4, "" );
    }
#endif

  return 0;
}
