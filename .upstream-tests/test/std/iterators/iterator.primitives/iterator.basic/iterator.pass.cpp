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

// <cuda/std/iterator>

// template<class Category, class T, class Distance = ptrdiff_t,
//          class Pointer = T*, class Reference = T&>
// struct iterator
// {
//   typedef T         value_type;
//   typedef Distance  difference_type;
//   typedef Pointer   pointer;
//   typedef Reference reference;
//   typedef Category  iterator_category;
// };

#include <hip/std/iterator>
#include <hip/std/type_traits>

#include "test_macros.h"

struct A {};

template <class T>
__host__ __device__
void
test2()
{
    typedef hip::std::iterator<hip::std::forward_iterator_tag, T> It;
    static_assert((hip::std::is_same<typename It::value_type, T>::value), "");
    static_assert((hip::std::is_same<typename It::difference_type, hip::std::ptrdiff_t>::value), "");
    static_assert((hip::std::is_same<typename It::pointer, T*>::value), "");
    static_assert((hip::std::is_same<typename It::reference, T&>::value), "");
    static_assert((hip::std::is_same<typename It::iterator_category, hip::std::forward_iterator_tag>::value), "");
}

template <class T>
__host__ __device__
void
test3()
{
    typedef hip::std::iterator<hip::std::bidirectional_iterator_tag, T, short> It;
    static_assert((hip::std::is_same<typename It::value_type, T>::value), "");
    static_assert((hip::std::is_same<typename It::difference_type, short>::value), "");
    static_assert((hip::std::is_same<typename It::pointer, T*>::value), "");
    static_assert((hip::std::is_same<typename It::reference, T&>::value), "");
    static_assert((hip::std::is_same<typename It::iterator_category, hip::std::bidirectional_iterator_tag>::value), "");
}

template <class T>
__host__ __device__
void
test4()
{
    typedef hip::std::iterator<hip::std::random_access_iterator_tag, T, int, const T*> It;
    static_assert((hip::std::is_same<typename It::value_type, T>::value), "");
    static_assert((hip::std::is_same<typename It::difference_type, int>::value), "");
    static_assert((hip::std::is_same<typename It::pointer, const T*>::value), "");
    static_assert((hip::std::is_same<typename It::reference, T&>::value), "");
    static_assert((hip::std::is_same<typename It::iterator_category, hip::std::random_access_iterator_tag>::value), "");
}

template <class T>
__host__ __device__
void
test5()
{
    typedef hip::std::iterator<hip::std::input_iterator_tag, T, long, const T*, const T&> It;
    static_assert((hip::std::is_same<typename It::value_type, T>::value), "");
    static_assert((hip::std::is_same<typename It::difference_type, long>::value), "");
    static_assert((hip::std::is_same<typename It::pointer, const T*>::value), "");
    static_assert((hip::std::is_same<typename It::reference, const T&>::value), "");
    static_assert((hip::std::is_same<typename It::iterator_category, hip::std::input_iterator_tag>::value), "");
}

int main(int, char**)
{
    test2<A>();
    test3<A>();
    test4<A>();
    test5<A>();

  return 0;
}
