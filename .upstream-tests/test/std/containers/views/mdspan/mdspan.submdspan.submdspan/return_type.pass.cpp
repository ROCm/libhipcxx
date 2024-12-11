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

constexpr auto dyn = hip::std::dynamic_extent;

//template<class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class ... SubArgs>

using submdspan_test_types = hip::std::tuple<
      // LayoutLeft to LayoutLeft
      hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,1>,hip::std::dextents<size_t,1>, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,1>,hip::std::dextents<size_t,1>, hip::std::pair<int,int>>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,1>,hip::std::dextents<size_t,0>, int>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,2>,hip::std::dextents<size_t,2>, hip::std::full_extent_t, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,2>,hip::std::dextents<size_t,2>, hip::std::full_extent_t, hip::std::pair<int,int>>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,2>,hip::std::dextents<size_t,1>, hip::std::full_extent_t, int>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,3>,hip::std::dextents<size_t,3>, hip::std::full_extent_t, hip::std::full_extent_t, hip::std::pair<int,int>>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,3>,hip::std::dextents<size_t,2>, hip::std::full_extent_t, hip::std::pair<int,int>, int>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,3>,hip::std::dextents<size_t,1>, hip::std::full_extent_t, int, int>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,3>,hip::std::dextents<size_t,1>, hip::std::pair<int,int>, int, int>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,6>,hip::std::dextents<size_t,3>, hip::std::full_extent_t, hip::std::full_extent_t, hip::std::pair<int,int>, int, int, int>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,6>,hip::std::dextents<size_t,2>, hip::std::full_extent_t, hip::std::pair<int,int>, int, int, int, int>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,6>,hip::std::dextents<size_t,1>, hip::std::full_extent_t, int, int, int ,int, int>
    , hip::std::tuple<hip::std::layout_left, hip::std::layout_left, hip::std::dextents<size_t,6>,hip::std::dextents<size_t,1>, hip::std::pair<int,int>, int, int, int, int, int>
    // LayoutRight to LayoutRight
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,1>,hip::std::dextents<size_t,1>, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,1>,hip::std::dextents<size_t,1>, hip::std::pair<int,int>>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,1>,hip::std::dextents<size_t,0>, int>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,2>,hip::std::dextents<size_t,2>, hip::std::full_extent_t, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,2>,hip::std::dextents<size_t,2>, hip::std::pair<int,int>, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,2>,hip::std::dextents<size_t,1>, int, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,3>,hip::std::dextents<size_t,3>, hip::std::pair<int,int>, hip::std::full_extent_t, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,3>,hip::std::dextents<size_t,2>, int, hip::std::pair<int,int>, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,3>,hip::std::dextents<size_t,1>, int, int, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,6>,hip::std::dextents<size_t,3>, int, int, int, hip::std::pair<int,int>, hip::std::full_extent_t, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,6>,hip::std::dextents<size_t,2>, int, int, int, int, hip::std::pair<int,int>, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::dextents<size_t,6>,hip::std::dextents<size_t,1>, int, int, int, int, int, hip::std::full_extent_t>
    // LayoutRight to LayoutRight Check Extents Preservation
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1>,hip::std::extents<size_t,1>, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1>,hip::std::extents<size_t,dyn>, hip::std::pair<int,int>>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1>,hip::std::extents<size_t>, int>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1,2>,hip::std::extents<size_t,1,2>, hip::std::full_extent_t, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1,2>,hip::std::extents<size_t,dyn,2>, hip::std::pair<int,int>, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1,2>,hip::std::extents<size_t,2>, int, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1,2,3>,hip::std::extents<size_t,dyn,2,3>, hip::std::pair<int,int>, hip::std::full_extent_t, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1,2,3>,hip::std::extents<size_t,dyn,3>, int, hip::std::pair<int,int>, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1,2,3>,hip::std::extents<size_t,3>, int, int, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1,2,3,4,5,6>,hip::std::extents<size_t,dyn,5,6>, int, int, int, hip::std::pair<int,int>, hip::std::full_extent_t, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1,2,3,4,5,6>,hip::std::extents<size_t,dyn,6>, int, int, int, int, hip::std::pair<int,int>, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_right, hip::std::extents<size_t,1,2,3,4,5,6>,hip::std::extents<size_t,6>, int, int, int, int, int, hip::std::full_extent_t>

    , hip::std::tuple<hip::std::layout_right, hip::std::layout_stride, hip::std::extents<size_t,1,2,3,4,5,6>,hip::std::extents<size_t,1,dyn,6>, hip::std::full_extent_t, int, hip::std::pair<int,int>, int, int, hip::std::full_extent_t>
    , hip::std::tuple<hip::std::layout_right, hip::std::layout_stride, hip::std::extents<size_t,1,2,3,4,5,6>,hip::std::extents<size_t,2,dyn,5>, int, hip::std::full_extent_t, hip::std::pair<int,int>, int, hip::std::full_extent_t, int>
    >;

template<class T> struct TestSubMDSpan;

template<class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class ... SubArgs>
struct TestSubMDSpan<
  hip::std::tuple<LayoutOrg,
             LayoutSub,
             ExtentsOrg,
             ExtentsSub,
             SubArgs...>>
{
    using mds_org_t = hip::std::mdspan<int, ExtentsOrg, LayoutOrg>;
    using mds_sub_t = hip::std::mdspan<int, ExtentsSub, LayoutSub>;
    using map_t = typename mds_org_t::mapping_type;

    using mds_sub_deduced_t = decltype(hip::std::submdspan(mds_org_t(nullptr, map_t()), SubArgs()...));
    using sub_args_t = hip::std::tuple<SubArgs...>;
};

// TYPED_TEST(TestSubMDSpan, submdspan_return_type)
template<class T>
__host__ __device__ void test_submdspan()
{
    using TestFixture = TestSubMDSpan<T>;

    static_assert(hip::std::is_same<typename TestFixture::mds_sub_t,
                                     typename TestFixture::mds_sub_deduced_t>::value,
                  "SubMDSpan: wrong return type");
}

int main(int, char**)
{
    static_assert( hip::std::tuple_size< submdspan_test_types >{} == 40, "" );

    test_submdspan< hip::std::tuple_element_t<  0, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t<  1, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t<  2, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t<  3, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t<  4, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t<  5, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t<  6, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t<  7, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t<  8, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t<  9, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 10, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 11, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 12, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 13, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 14, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 15, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 16, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 17, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 18, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 19, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 20, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 21, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 22, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 23, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 24, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 25, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 26, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 27, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 28, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 29, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 30, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 31, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 32, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 33, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 34, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 35, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 36, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 37, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 38, submdspan_test_types > >();
    test_submdspan< hip::std::tuple_element_t< 39, submdspan_test_types > >();

    return 0;
}
