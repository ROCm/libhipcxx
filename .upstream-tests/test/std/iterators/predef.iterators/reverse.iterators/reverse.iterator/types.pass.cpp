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

// reverse_iterator

// Test nested types and data member:

// template <BidirectionalIterator Iter>
// class reverse_iterator {
// protected:
//   Iter current;
// public:
//   iterator<typename iterator_traits<Iterator>::iterator_category,
//   typename iterator_traits<Iterator>::value_type,
//   typename iterator_traits<Iterator>::difference_type,
//   typename iterator_traits<Iterator>::pointer,
//   typename iterator_traits<Iterator>::reference> {
// };

#include <hip/std/iterator>
#include <hip/std/type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
struct find_current
    : private hip::std::reverse_iterator<It>
{
__host__ __device__
    void test() {++(this->current);}
};

template <class It>
__host__ __device__
void
test()
{
    typedef hip::std::reverse_iterator<It> R;
    typedef hip::std::iterator_traits<It> T;
    find_current<It> q;
    q.test();
    static_assert((hip::std::is_same<typename R::iterator_type, It>::value), "");
    static_assert((hip::std::is_same<typename R::value_type, typename T::value_type>::value), "");
    static_assert((hip::std::is_same<typename R::difference_type, typename T::difference_type>::value), "");
    static_assert((hip::std::is_same<typename R::reference, typename T::reference>::value), "");
    static_assert((hip::std::is_same<typename R::pointer, typename hip::std::iterator_traits<It>::pointer>::value), "");
    static_assert((hip::std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
}

int main(int, char**)
{
    test<bidirectional_iterator<char*> >();
    test<random_access_iterator<char*> >();
    test<char*>();

  return 0;
}
