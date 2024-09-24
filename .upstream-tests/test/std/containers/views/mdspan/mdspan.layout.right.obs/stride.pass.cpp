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
    using ext0d_t = hip::std::extents<index_t>;
    using ext2d_t = hip::std::extents<index_t,dyn,dyn>;

    {
        ext2d_t e{64, 128};
        hip::std::layout_right::mapping<ext2d_t> m{ e };

        assert( m.stride(0) == 128 );
        assert( m.stride(1) ==   1 );

        static_assert( is_stride_avail_v< decltype(m), index_t > == true , "" );
    }

    {
        ext2d_t e{64, 1};
        hip::std::layout_right::mapping<ext2d_t> m{ e };

        assert( m.stride(0) == 1 );
        assert( m.stride(1) == 1 );
    }

    // constraint: extents_type::rank() > 0
    {
        ext0d_t e{};
        hip::std::layout_right::mapping<ext0d_t> m{ e };

        unused( m );

        static_assert( is_stride_avail_v< decltype(m), index_t > == false, "" );
    }

    return 0;
}
