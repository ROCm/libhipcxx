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
#include "../mdspan.mdspan.util/mdspan_util.hpp"
#include "../foo_customizations.hpp"

constexpr auto dyn = hip::std::dynamic_extent;

int main(int, char**)
{
    {
        using  data_t = int;
        using   acc_t = Foo::foo_accessor<data_t>;
        using index_t = size_t;

        hip::std::array<data_t, 1> d{42};
        hip::std::layout_left::mapping<hip::std::extents<index_t,dyn,dyn>> map{hip::std::dextents<index_t,2>{64, 128}};
        acc_t a;
        hip::std::mdspan<data_t, hip::std::extents<index_t,dyn,dyn>, hip::std::layout_left, acc_t> m{ d.data(), map, a };

        static_assert(m.is_exhaustive(), "");
        //assert(m.data_handle()  == d.data());
        assert(m.rank()         == 2       );
        assert(m.rank_dynamic() == 2       );
        assert(m.extent(0)      == 64      );
        assert(m.extent(1)      == 128     );
    }

    return 0;
}
