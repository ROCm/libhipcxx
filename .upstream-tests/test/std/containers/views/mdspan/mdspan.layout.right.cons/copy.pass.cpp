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

int main(int, char**)
{
    using index_t = size_t;

    {

        hip::std::layout_right::mapping<hip::std::extents<index_t,dyn,dyn>> m0{hip::std::dextents<index_t,2>{16,32}};
        hip::std::layout_right::mapping<hip::std::extents<index_t,dyn,dyn>> m {m0};

        static_assert( m.is_exhaustive()          == true, "" );
        static_assert( m.extents().rank()         == 2   , "" );
        static_assert( m.extents().rank_dynamic() == 2   , "" );

        assert( m.extents().extent(0) == 16 );
        assert( m.extents().extent(1) == 32 );
        assert( m.stride(0)           == 32 );
        assert( m.stride(1)           == 1  );
    }

    // Constraint: is_constructible_v<extents_type, OtherExtents> is true
    {
        using mapping0_t = hip::std::layout_right::mapping<hip::std::extents <index_t,16,32>>;
        using mapping1_t = hip::std::layout_right::mapping<hip::std::extents <index_t,16,16>>;
        using mappingd_t = hip::std::layout_right::mapping<hip::std::dextents<index_t,2>>;

        static_assert( is_cons_avail_v< mappingd_t, mapping0_t > == true , "" );
        static_assert( is_cons_avail_v< mapping1_t, mapping0_t > == false, "" );
    }

    return 0;
}