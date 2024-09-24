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
#include <hip/std/array>
#include <hip/std/cassert>

#define CHECK_MAPPING(m) \
        assert( m.is_exhaustive()          == false); \
        assert( m.extents().rank()         == 2    ); \
        assert( m.extents().rank_dynamic() == 2    ); \
        assert( m.extents().extent(0)      == 16   ); \
        assert( m.extents().extent(1)      == 32   ); \
        assert( m.stride(0)                == 1    ); \
        assert( m.stride(1)                == 128  ); \
        assert( m.strides()[0]             == 1    ); \
        assert( m.strides()[1]             == 128  )

constexpr auto dyn = hip::std::dynamic_extent;

int main(int, char**)
{
    // From a span
    {
        typedef int    data_t ;
        typedef size_t index_t;

        using my_ext = typename hip::std::extents<size_t,dyn>;

        hip::std::array<int,2> a{1, 128};
        hip::std::span <int,2> s(a.data(), 2);
        hip::std::layout_stride::mapping<hip::std::extents<size_t,dyn, dyn>> m{hip::std::dextents<size_t,2>{16, 32}, s};

        CHECK_MAPPING(m);
    }

    // TEST(TestLayoutStrideListInitialization, test_list_initialization)
    {
        typedef int    data_t ;
        typedef size_t index_t;

        hip::std::layout_stride::mapping<hip::std::extents<size_t,dyn, dyn>> m{hip::std::dextents<size_t,2>{16, 32}, hip::std::array<int,2>{1, 128}};

        CHECK_MAPPING(m);
    }

    // From another mapping
    {
        typedef int    data_t ;
        typedef size_t index_t;

        hip::std::layout_stride::mapping<hip::std::extents<index_t,dyn, dyn>> m0{hip::std::dextents<index_t,2>{16, 32}, hip::std::array<int,2>{1, 128}};
        hip::std::layout_stride::mapping<hip::std::extents<index_t,dyn, dyn>> m{m0};

        CHECK_MAPPING(m);
    }

    return 0;
}
