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

//UNSUPPORTED: c++11, nvrtc && nvcc-12.0, nvrtc && nvcc-12.1

#include <hip/std/mdspan>
#include <hip/std/cassert>
#include "../mdspan.extents.util/extents_util.hpp"
#include "../my_int.hpp"

// TYPED_TEST(TestExtents, array_ctor)
template<class T>
__host__ __device__ void test_array_con()
{
    using TestFixture = TestExtents<T>;
    TestFixture t;

    auto e = typename TestFixture::extents_type(t.dyn_sizes);
    assert(e == t.exts);
}

template< class T, class IndexType, size_t N, class = void >
struct is_array_cons_avail : hip::std::false_type {};

template< class T, class IndexType, size_t N >
struct is_array_cons_avail< T
                          , IndexType
                          , N
                          , hip::std::enable_if_t< hip::std::is_same< decltype( T{ hip::std::declval<hip::std::array<IndexType, N>>() } )
                                                                      , T
                                                                      >::value
                                                  >
                          > : hip::std::true_type {};

template< class T, class IndexType, size_t N >
constexpr bool is_array_cons_avail_v = is_array_cons_avail< T, IndexType, N >::value;

int main(int, char**)
{
    test_array_con< hip::std::tuple_element_t< 0, extents_test_types > >();
    test_array_con< hip::std::tuple_element_t< 1, extents_test_types > >();
    test_array_con< hip::std::tuple_element_t< 2, extents_test_types > >();
    test_array_con< hip::std::tuple_element_t< 3, extents_test_types > >();
    test_array_con< hip::std::tuple_element_t< 4, extents_test_types > >();
    test_array_con< hip::std::tuple_element_t< 5, extents_test_types > >();

    static_assert( is_array_cons_avail_v< hip::std::dextents<   int,2>, int   , 2 > == true , "" );

    static_assert( is_array_cons_avail_v< hip::std::dextents<   int,2>, my_int, 2 > == true , "" );

    // Constraint: rank consistency
    static_assert( is_array_cons_avail_v< hip::std::dextents<   int,1>, int   , 2 > == false, "" );

    // Constraint: convertibility
    static_assert( is_array_cons_avail_v< hip::std::dextents<my_int,1>, my_int_non_convertible          , 1 > == false, "" );

    // Constraint: nonthrow-constructibility
    static_assert( is_array_cons_avail_v< hip::std::dextents<   int,1>, my_int_non_nothrow_constructible, 1 > == false, "" );

    return 0;
}
