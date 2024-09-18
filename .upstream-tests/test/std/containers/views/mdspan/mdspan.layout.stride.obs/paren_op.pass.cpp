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
#include "../my_int.hpp"
#include "../mdspan.layout.util/layout_util.hpp"

constexpr auto dyn = hip::std::dynamic_extent;

int main(int, char**)
{
    using index_t = int;

    {
        hip::std::extents<index_t,16> e;
        hip::std::array<index_t,1> a{1};
        hip::std::layout_stride::mapping<hip::std::extents<index_t,16>> m{e, a};

        assert( m(8) == 8 );
    }

    {
        hip::std::extents<index_t,dyn,dyn> e{16, 32};
        hip::std::array<index_t,2> a{1,16};
        hip::std::layout_stride::mapping<hip::std::extents<index_t,dyn,dyn>> m{e, a};

        assert( m(8,16) == 8*1 + 16*16 );
    }

    {
        hip::std::extents<index_t,16,dyn> e{32};
        hip::std::array<index_t,2> a{1,24};
        hip::std::layout_stride::mapping<hip::std::extents<index_t,dyn,dyn>> m{e, a};

        assert( m(8,16) == 8*1 + 16*24 );
    }

    {
        hip::std::extents<index_t,16,dyn> e{32};
        hip::std::array<index_t,2> a{48,1};
        hip::std::layout_stride::mapping<hip::std::extents<index_t,dyn,dyn>> m{e, a};

        assert( m(8,16) == 8*48 + 16*1 );
    }

    // Indices are of a type implicitly convertible to index_type
    {
        hip::std::extents<index_t,dyn,dyn> e{16, 32};
        hip::std::array<index_t,2> a{1,16};
        hip::std::layout_stride::mapping<hip::std::extents<index_t,dyn,dyn>> m{e, a};

        assert( m(my_int(8),my_int(16)) == 8*1 + 16*16 );
    }

    // Constraints
    {
        hip::std::extents<index_t,16> e;
        hip::std::array<index_t,1> a{1};
        hip::std::layout_stride::mapping<hip::std::extents<index_t,16>> m{e, a};

        static_assert( is_paren_op_avail_v< decltype(m), index_t          > ==  true, "" );

        // rank consistency
        static_assert( is_paren_op_avail_v< decltype(m), index_t, index_t > == false, "" );

        // convertibility
        static_assert( is_paren_op_avail_v< decltype(m), my_int_non_convertible           > == false, "" );

        // nothrow-constructibility
        static_assert( is_paren_op_avail_v< decltype(m), my_int_non_nothrow_constructible > == false, "" );
    }

    return 0;
}

