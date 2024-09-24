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
#include <hip/std/tuple>
#include <hip/std/utility>
#include "../mdspan.layout.util/layout_util.hpp"

constexpr auto dyn = hip::std::dynamic_extent;

template <class Extents, size_t... DynamicSizes>
using test_left_type = hip::std::tuple<
    typename hip::std::layout_left::template mapping<Extents>,
    hip::std::integer_sequence<size_t, DynamicSizes...>
>;

__host__ __device__ void typed_test_default_ctor_left()
{
    typed_test_default_ctor< test_left_type< hip::std::extents<size_t,10     >     > >();
    typed_test_default_ctor< test_left_type< hip::std::extents<size_t,dyn    >, 10 > >();
    typed_test_default_ctor< test_left_type< hip::std::extents<size_t,dyn, 10>,  5 > >();
    typed_test_default_ctor< test_left_type< hip::std::extents<size_t,  5,dyn>, 10 > >();
    typed_test_default_ctor< test_left_type< hip::std::extents<size_t,  5, 10>     > >();
}

__host__ __device__ void typed_test_compatible_left()
{
    typed_test_compatible< test_left_type_pair<_exts<dyn         >, _sizes<10    >, _exts< 10          >, _sizes<          >> >();
    typed_test_compatible< test_left_type_pair<_exts<dyn,  10    >, _sizes< 5    >, _exts<  5, dyn     >, _sizes<10        >> >();
    typed_test_compatible< test_left_type_pair<_exts<dyn, dyn    >, _sizes< 5, 10>, _exts<  5, dyn     >, _sizes<10        >> >();
    typed_test_compatible< test_left_type_pair<_exts<dyn, dyn    >, _sizes< 5, 10>, _exts<dyn,  10     >, _sizes< 5        >> >();
    typed_test_compatible< test_left_type_pair<_exts<dyn, dyn    >, _sizes< 5, 10>, _exts<  5,  10     >, _sizes<          >> >();
    typed_test_compatible< test_left_type_pair<_exts<  5,  10    >, _sizes<      >, _exts<  5, dyn     >, _sizes<10        >> >();
    typed_test_compatible< test_left_type_pair<_exts<  5,  10    >, _sizes<      >, _exts<dyn,  10     >, _sizes< 5        >> >();
    typed_test_compatible< test_left_type_pair<_exts<dyn, dyn, 15>, _sizes< 5, 10>, _exts<  5, dyn, 15 >, _sizes<10        >> >();
    typed_test_compatible< test_left_type_pair<_exts<  5,  10, 15>, _sizes<      >, _exts<  5, dyn, 15 >, _sizes<10        >> >();
    typed_test_compatible< test_left_type_pair<_exts<  5,  10, 15>, _sizes<      >, _exts<dyn, dyn, dyn>, _sizes< 5, 10, 15>> >();
}

int main(int, char**)
{
    typed_test_default_ctor_left();

    typed_test_compatible_left();

    // TEST(TestLayoutLeftListInitialization, test_layout_left_extent_initialization)
    {
        typedef int    data_t ;
        typedef size_t index_t;

        hip::std::layout_left::mapping<hip::std::extents<size_t,dyn, dyn>> m{hip::std::dextents<size_t,2>{16, 32}};

        static_assert( m.is_exhaustive()          == true, "" );
        static_assert( m.extents().rank()         == 2   , "" );
        static_assert( m.extents().rank_dynamic() == 2   , "" );

        assert( m.extents().extent(0) == 16 );
        assert( m.extents().extent(1) == 32 );
        assert( m.stride(0)           == 1  );
        assert( m.stride(1)           == 16 );
    }

    return 0;
}
