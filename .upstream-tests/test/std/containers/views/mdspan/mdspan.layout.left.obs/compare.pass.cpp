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
#include "../mdspan.layout.util/layout_util.hpp"

constexpr auto dyn = hip::std::dynamic_extent;

__host__ __device__ void typed_test_compare_left()
{
    typed_test_compare< test_left_type_pair<_exts<dyn         >, _sizes<10    >, _exts< 10          >, _sizes<          >> >();
    typed_test_compare< test_left_type_pair<_exts<dyn,  10    >, _sizes< 5    >, _exts<  5, dyn     >, _sizes<10        >> >();
    typed_test_compare< test_left_type_pair<_exts<dyn, dyn    >, _sizes< 5, 10>, _exts<  5, dyn     >, _sizes<10        >> >();
    typed_test_compare< test_left_type_pair<_exts<dyn, dyn    >, _sizes< 5, 10>, _exts<dyn,  10     >, _sizes< 5        >> >();
    typed_test_compare< test_left_type_pair<_exts<dyn, dyn    >, _sizes< 5, 10>, _exts<  5,  10     >, _sizes<          >> >();
    typed_test_compare< test_left_type_pair<_exts<  5,  10    >, _sizes<      >, _exts<  5, dyn     >, _sizes<10        >> >();
    typed_test_compare< test_left_type_pair<_exts<  5,  10    >, _sizes<      >, _exts<dyn,  10     >, _sizes< 5        >> >();
    typed_test_compare< test_left_type_pair<_exts<dyn, dyn, 15>, _sizes< 5, 10>, _exts<  5, dyn, 15 >, _sizes<10        >> >();
    typed_test_compare< test_left_type_pair<_exts<  5,  10, 15>, _sizes<      >, _exts<  5, dyn, 15 >, _sizes<10        >> >();
    typed_test_compare< test_left_type_pair<_exts<  5,  10, 15>, _sizes<      >, _exts<dyn, dyn, dyn>, _sizes< 5, 10, 15>> >();
}

int main(int, char**)
{
    typed_test_compare_left();

    using index_t = size_t;
    using ext1d_t = hip::std::extents<index_t,dyn>;
    using ext2d_t = hip::std::extents<index_t,dyn,dyn>;

    {
        ext2d_t e{64, 128};
        hip::std::layout_left::mapping<ext2d_t> m0{  e };
        hip::std::layout_left::mapping<ext2d_t> m { m0 };

        assert( m == m0 );
    }

    {
        ext2d_t e0{64, 128};
        ext2d_t e1{16,  32};
        hip::std::layout_left::mapping<ext2d_t> m0{ e0 };
        hip::std::layout_left::mapping<ext2d_t> m1{ e1 };

        assert( m0 != m1 );
    }

    return 0;
}
