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

// front_insert_iterator

// Test nested types and data member:

// template <class Container>
// class front_insert_iterator {
// protected:
//   Container* container;
// public:
//   typedef Container                   container_type;
//   typedef void                        value_type;
//   typedef void                        difference_type;
//   typedef void                        reference;
//   typedef void                        pointer;
//   typedef output_iterator_tag         iterator_category;
// };

#include <hip/std/iterator>
#include <hip/std/type_traits>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#include <hip/std/vector>

#include "test_macros.h"

template <class C>
struct find_container
    : private hip::std::front_insert_iterator<C>
{
__host__ __device__
    explicit find_container(C& c) : hip::std::front_insert_iterator<C>(c) {}
__host__ __device__
    void test() {this->container = 0;}
};

template <class C>
__host__ __device__
void
test()
{
    typedef hip::std::front_insert_iterator<C> R;
    C c;
    find_container<C> q(c);
    q.test();
    static_assert((hip::std::is_same<typename R::container_type, C>::value), "");
    static_assert((hip::std::is_same<typename R::value_type, void>::value), "");
    static_assert((hip::std::is_same<typename R::difference_type, void>::value), "");
    static_assert((hip::std::is_same<typename R::reference, void>::value), "");
    static_assert((hip::std::is_same<typename R::pointer, void>::value), "");
    static_assert((hip::std::is_same<typename R::iterator_category, hip::std::output_iterator_tag>::value), "");
}

int main(int, char**)
{
    test<hip::std::vector<int> >();

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
