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

// <cuda/std/array>

// template <class T, size_t N >
// struct array
// {
//     // types:
//     typedef T& reference;
//     typedef const T& const_reference;
//     typedef implementation defined iterator;
//     typedef implementation defined const_iterator;
//     typedef T value_type;
//     typedef T* pointer;
//     typedef size_t size_type;
//     typedef ptrdiff_t difference_type;
//     typedef T value_type;
//     typedef hip::std::reverse_iterator<cuda/std/iterator> reverse_iterator;
//     typedef hip::std::reverse_iterator<const_iterator> const_reverse_iterator;

#include <hip/std/array>
#include <hip/std/iterator>
#include <hip/std/type_traits>

#include "test_macros.h"

template <class C>
__host__ __device__
void test_iterators() {
    typedef hip::std::iterator_traits<typename C::iterator> ItT;
    typedef hip::std::iterator_traits<typename C::const_iterator> CItT;
    static_assert((hip::std::is_same<typename ItT::iterator_category, hip::std::random_access_iterator_tag>::value), "");
    static_assert((hip::std::is_same<typename ItT::value_type, typename C::value_type>::value), "");
    static_assert((hip::std::is_same<typename ItT::reference, typename C::reference>::value), "");
    static_assert((hip::std::is_same<typename ItT::pointer, typename C::pointer>::value), "");
    static_assert((hip::std::is_same<typename ItT::difference_type, typename C::difference_type>::value), "");

    static_assert((hip::std::is_same<typename CItT::iterator_category, hip::std::random_access_iterator_tag>::value), "");
    static_assert((hip::std::is_same<typename CItT::value_type, typename C::value_type>::value), "");
    static_assert((hip::std::is_same<typename CItT::reference, typename C::const_reference>::value), "");
    static_assert((hip::std::is_same<typename CItT::pointer, typename C::const_pointer>::value), "");
    static_assert((hip::std::is_same<typename CItT::difference_type, typename C::difference_type>::value), "");
}

int main(int, char**)
{
    {
        typedef double T;
        typedef hip::std::array<T, 10> C;
        static_assert((hip::std::is_same<C::reference, T&>::value), "");
        static_assert((hip::std::is_same<C::const_reference, const T&>::value), "");
        LIBCPP_STATIC_ASSERT((hip::std::is_same<C::iterator, T*>::value), "");
        LIBCPP_STATIC_ASSERT((hip::std::is_same<C::const_iterator, const T*>::value), "");
        test_iterators<C>();
        static_assert((hip::std::is_same<C::pointer, T*>::value), "");
        static_assert((hip::std::is_same<C::const_pointer, const T*>::value), "");
        static_assert((hip::std::is_same<C::size_type, hip::std::size_t>::value), "");
        static_assert((hip::std::is_same<C::difference_type, hip::std::ptrdiff_t>::value), "");
        static_assert((hip::std::is_same<C::reverse_iterator, hip::std::reverse_iterator<C::iterator> >::value), "");
        static_assert((hip::std::is_same<C::const_reverse_iterator, hip::std::reverse_iterator<C::const_iterator> >::value), "");

        static_assert((hip::std::is_signed<typename C::difference_type>::value), "");
        static_assert((hip::std::is_unsigned<typename C::size_type>::value), "");
        static_assert((hip::std::is_same<typename C::difference_type,
            typename hip::std::iterator_traits<typename C::iterator>::difference_type>::value), "");
        static_assert((hip::std::is_same<typename C::difference_type,
            typename hip::std::iterator_traits<typename C::const_iterator>::difference_type>::value), "");
    }
    {
        typedef int* T;
        typedef hip::std::array<T, 0> C;
        static_assert((hip::std::is_same<C::reference, T&>::value), "");
        static_assert((hip::std::is_same<C::const_reference, const T&>::value), "");
        LIBCPP_STATIC_ASSERT((hip::std::is_same<C::iterator, T*>::value), "");
        LIBCPP_STATIC_ASSERT((hip::std::is_same<C::const_iterator, const T*>::value), "");
        test_iterators<C>();
        static_assert((hip::std::is_same<C::pointer, T*>::value), "");
        static_assert((hip::std::is_same<C::const_pointer, const T*>::value), "");
        static_assert((hip::std::is_same<C::size_type, hip::std::size_t>::value), "");
        static_assert((hip::std::is_same<C::difference_type, hip::std::ptrdiff_t>::value), "");
        static_assert((hip::std::is_same<C::reverse_iterator, hip::std::reverse_iterator<C::iterator> >::value), "");
        static_assert((hip::std::is_same<C::const_reverse_iterator, hip::std::reverse_iterator<C::const_iterator> >::value), "");

        static_assert((hip::std::is_signed<typename C::difference_type>::value), "");
        static_assert((hip::std::is_unsigned<typename C::size_type>::value), "");
        static_assert((hip::std::is_same<typename C::difference_type,
            typename hip::std::iterator_traits<typename C::iterator>::difference_type>::value), "");
        static_assert((hip::std::is_same<typename C::difference_type,
            typename hip::std::iterator_traits<typename C::const_iterator>::difference_type>::value), "");
    }

  return 0;
}
