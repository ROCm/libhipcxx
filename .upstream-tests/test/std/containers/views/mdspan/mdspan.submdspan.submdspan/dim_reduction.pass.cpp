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

int main(int, char**)
{
    // TEST(TestSubmdspanLayoutRightStaticSizedRankReducing3Dto1D, test_submdspan_layout_right_static_sized_rank_reducing_3d_to_1d)
    {
        hip::std::array<int,2*3*4> d;
        hip::std::mdspan<int, hip::std::extents<size_t,2, 3, 4>> m(d.data());
        m(1, 1, 1) = 42;
        auto sub0 = hip::std::submdspan(m, 1, 1, hip::std::full_extent);

        static_assert(decltype(sub0)::rank()==1,"unexpected submdspan rank");
        static_assert(sub0.rank()         ==  1, "");
        static_assert(sub0.rank_dynamic() ==  0, "");
        assert(sub0.extent(0) ==  4);
        assert(sub0(1)        == 42);
    }

    // TEST(TestSubmdspanLayoutLeftStaticSizedRankReducing3Dto1D, test_submdspan_layout_left_static_sized_rank_reducing_3d_to_1d)
    {
        hip::std::array<int,2*3*4> d;
        hip::std::mdspan<int, hip::std::extents<size_t,2, 3, 4>, hip::std::layout_left> m(d.data());
        m(1, 1, 1) = 42;
        auto sub0 = hip::std::submdspan(m, 1, 1, hip::std::full_extent);

        static_assert(sub0.rank()         ==  1, "");
        static_assert(sub0.rank_dynamic() ==  0, "");
        assert(sub0.extent(0) ==  4);
        assert(sub0(1)        == 42);
    }

    // TEST(TestSubmdspanLayoutRightStaticSizedRankReducingNested3Dto0D, test_submdspan_layout_right_static_sized_rank_reducing_nested_3d_to_0d)
    {
        hip::std::array<int,2*3*4> d;
        hip::std::mdspan<int, hip::std::extents<size_t,2, 3, 4>> m(d.data());
        m(1, 1, 1) = 42;
        auto sub0 = hip::std::submdspan(m, 1, hip::std::full_extent, hip::std::full_extent);

        static_assert(sub0.rank()         == 2, "");
        static_assert(sub0.rank_dynamic() == 0, "");
        assert(sub0.extent(0) ==  3);
        assert(sub0.extent(1) ==  4);
        assert(sub0(1, 1)     == 42);

        auto sub1 = hip::std::submdspan(sub0, 1, hip::std::full_extent);
        static_assert(sub1.rank()         == 1, "");
        static_assert(sub1.rank_dynamic() == 0, "");
        assert(sub1.extent(0) ==  4);
        assert(sub1(1)        == 42);

        auto sub2 = hip::std::submdspan(sub1, 1);
        static_assert(sub2.rank()         == 0, "");
        static_assert(sub2.rank_dynamic() == 0, "");
        assert(sub2() == 42);
    }

    return 0;
}




