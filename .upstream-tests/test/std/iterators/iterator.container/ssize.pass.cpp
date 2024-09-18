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

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <cuda/std/iterator>
// template <class C> constexpr auto ssize(const C& c)
//     -> common_type_t<ptrdiff_t, make_signed_t<decltype(c.size())>>;                    // C++20
// template <class T, ptrdiff_t> constexpr ptrdiff_t ssize(const T (&array)[N]) noexcept; // C++20

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
#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#include <hip/std/string_view>
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif

#include "test_macros.h"

struct short_container {
__host__ __device__
    uint16_t size() const { return 60000; } // not noexcept
};


template<typename C>
__host__ __device__
void test_container(C& c)
{
//  Can't say noexcept here because the container might not be
    static_assert( hip::std::is_signed_v<decltype(hip::std::ssize(c))>, "");
    assert ( hip::std::ssize(c)   == static_cast<decltype(hip::std::ssize(c))>(c.size()));
}

template<typename C>
__host__ __device__
void test_const_container(const C& c)
{
//  Can't say noexcept here because the container might not be
    static_assert( hip::std::is_signed_v<decltype(hip::std::ssize(c))>, "");
    assert ( hip::std::ssize(c)   == static_cast<decltype(hip::std::ssize(c))>(c.size()));
}

template<typename T>
__host__ __device__
void test_const_container(const hip::std::initializer_list<T>& c)
{
    LIBCPP_ASSERT_NOEXCEPT(hip::std::ssize(c)); // our hip::std::ssize is conditionally noexcept
    static_assert( hip::std::is_signed_v<decltype(hip::std::ssize(c))>, "");
    assert ( hip::std::ssize(c)   == static_cast<decltype(hip::std::ssize(c))>(c.size()));
}

template<typename T>
__host__ __device__
void test_container(hip::std::initializer_list<T>& c)
{
    LIBCPP_ASSERT_NOEXCEPT(hip::std::ssize(c)); // our hip::std::ssize is conditionally noexcept
    static_assert( hip::std::is_signed_v<decltype(hip::std::ssize(c))>, "");
    assert ( hip::std::ssize(c)   == static_cast<decltype(hip::std::ssize(c))>(c.size()));
}

template<typename T, size_t Sz>
__host__ __device__
void test_const_array(const T (&array)[Sz])
{
    ASSERT_NOEXCEPT(hip::std::ssize(array));
    static_assert( hip::std::is_signed_v<decltype(hip::std::ssize(array))>, "");
    assert ( hip::std::ssize(array) == Sz );
}

__device__ static constexpr int arrA [] { 1, 2, 3 };

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_VECTOR)
    hip::std::vector<int> v; v.push_back(1);
#endif
#if defined(_LIBCUDACXX_HAS_LIST)
    hip::std::list<int>   l; l.push_back(2);
#endif
    hip::std::array<int, 1> a; a[0] = 3;
    hip::std::initializer_list<int> il = { 4 };

#if defined(_LIBCUDACXX_HAS_VECTOR)
    test_container ( v );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(hip::std::ssize(v)));
#endif
#if defined(_LIBCUDACXX_HAS_LIST)
    test_container ( l );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(hip::std::ssize(l)));
#endif
    test_container ( a );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(hip::std::ssize(a)));
    test_container ( il );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(hip::std::ssize(il)));

#if defined(_LIBCUDACXX_HAS_VECTOR)
    test_const_container ( v );
#endif
#if defined(_LIBCUDACXX_HAS_LIST)
    test_const_container ( l );
#endif
    test_const_container ( a );
    test_const_container ( il );

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
    hip::std::string_view sv{"ABC"};
    test_container ( sv );
    ASSERT_SAME_TYPE(ptrdiff_t, decltype(hip::std::ssize(sv)));
    test_const_container ( sv );
#endif

    ASSERT_SAME_TYPE(ptrdiff_t, decltype(hip::std::ssize(arrA)));
    static_assert( hip::std::is_signed_v<decltype(hip::std::ssize(arrA))>, "");
    test_const_array ( arrA );

//  From P1227R2:
//     Note that the code does not just return the hip::std::make_signed variant of
//     the container's size() method, because it's conceivable that a container
//     might choose to represent its size as a uint16_t, supporting up to
//     65,535 elements, and it would be a disaster for hip::std::ssize() to turn a
//     size of 60,000 into a size of -5,536.

    short_container sc;
//  is the return type signed? Is it big enough to hold 60K?
//  is the "signed version" of sc.size() too small?
    static_assert( hip::std::is_signed_v<                      decltype(hip::std::ssize(sc))>, "");
    static_assert( hip::std::numeric_limits<                   decltype(hip::std::ssize(sc))>::max()  > 60000, "");
    static_assert( hip::std::numeric_limits<hip::std::make_signed_t<decltype(hip::std:: size(sc))>>::max() < 60000, "");
#ifdef __CUDA_ARCH__
    assert (hip::std::ssize(sc) == 60000);
#endif
    LIBCPP_ASSERT_NOT_NOEXCEPT(hip::std::ssize(sc));

  return 0;
}
