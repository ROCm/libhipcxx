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

// TYPED_TEST(TestExtents, copy_ctor)
template<class T>
__host__ __device__ void test_copy_con()
{
    using TestFixture = TestExtents<T>;
    TestFixture t;

    typename TestFixture::extents_type e { t.exts };
    assert(e == t.exts);
}

template< class T1, class T2, class = void >
struct is_copy_cons_avail : hip::std::false_type {};

template< class T1, class T2 >
struct is_copy_cons_avail< T1
                         , T2
                         , hip::std::enable_if_t< hip::std::is_same< decltype( T1{ hip::std::declval<T2>() } )
                                                                     , T1
                                                                     >::value
                                                 >
                         > : hip::std::true_type {};

template< class T1, class T2 >
constexpr bool is_copy_cons_avail_v = is_copy_cons_avail< T1, T2 >::value;

int main(int, char**)
{
    test_copy_con< hip::std::tuple_element_t< 0, extents_test_types > >();
    test_copy_con< hip::std::tuple_element_t< 1, extents_test_types > >();
    test_copy_con< hip::std::tuple_element_t< 2, extents_test_types > >();
    test_copy_con< hip::std::tuple_element_t< 3, extents_test_types > >();
    test_copy_con< hip::std::tuple_element_t< 4, extents_test_types > >();
    test_copy_con< hip::std::tuple_element_t< 5, extents_test_types > >();

    static_assert( is_copy_cons_avail_v< hip::std::extents<int,2  >, hip::std::extents<int,2> > == true , "" );

    // Constraint: rank consistency
    static_assert( is_copy_cons_avail_v< hip::std::extents<int,2,2>, hip::std::extents<int,2> > == false, "" );

    // Constraint: extents consistency
    static_assert( is_copy_cons_avail_v< hip::std::extents<int,1  >, hip::std::extents<int,2> > == false, "" );

    return 0;
}
