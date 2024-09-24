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

// move_iterator

// Test nested types:

// template <InputIterator Iter>
// class move_iterator {
// public:
//   typedef Iter                  iterator_type;
//   typedef Iter::difference_type difference_type;
//   typedef Iter                  pointer;
//   typedef Iter::value_type      value_type;
//   typedef value_type&&          reference;
// };

#include <hip/std/iterator>
#include <hip/std/type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class ValueType, class Reference>
struct DummyIt {
  typedef hip::std::forward_iterator_tag iterator_category;
  typedef ValueType value_type;
  typedef hip::std::ptrdiff_t difference_type;
  typedef ValueType* pointer;
  typedef Reference reference;
};

template <class It>
__host__ __device__
void
test()
{
    typedef hip::std::move_iterator<It> R;
    typedef hip::std::iterator_traits<It> T;
    static_assert((hip::std::is_same<typename R::iterator_type, It>::value), "");
    static_assert((hip::std::is_same<typename R::difference_type, typename T::difference_type>::value), "");
    static_assert((hip::std::is_same<typename R::pointer, It>::value), "");
    static_assert((hip::std::is_same<typename R::value_type, typename T::value_type>::value), "");
#if TEST_STD_VER >= 11
    static_assert((hip::std::is_same<typename R::reference, typename R::value_type&&>::value), "");
#else
    static_assert((hip::std::is_same<typename R::reference, typename T::reference>::value), "");
#endif
    static_assert((hip::std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
}

int main(int, char**)
{
    test<input_iterator<char*> >();
    test<forward_iterator<char*> >();
    test<bidirectional_iterator<char*> >();
    test<random_access_iterator<char*> >();
    test<char*>();
#if TEST_STD_VER >= 11
    {
        typedef DummyIt<int, int> T;
        typedef hip::std::move_iterator<T> It;
        static_assert(hip::std::is_same<It::reference, int>::value, "");
    }
    {
        typedef DummyIt<int, hip::std::reference_wrapper<int> > T;
        typedef hip::std::move_iterator<T> It;
        static_assert(hip::std::is_same<It::reference, hip::std::reference_wrapper<int> >::value, "");
    }
    {
        // Check that move_iterator uses whatever reference type it's given
        // when it's not a reference.
        typedef DummyIt<int, long > T;
        typedef hip::std::move_iterator<T> It;
        static_assert(hip::std::is_same<It::reference, long>::value, "");
    }
    {
        typedef DummyIt<int, int&> T;
        typedef hip::std::move_iterator<T> It;
        static_assert(hip::std::is_same<It::reference, int&&>::value, "");
    }
    {
        typedef DummyIt<int, int&&> T;
        typedef hip::std::move_iterator<T> It;
        static_assert(hip::std::is_same<It::reference, int&&>::value, "");
    }
#endif

  return 0;
}
