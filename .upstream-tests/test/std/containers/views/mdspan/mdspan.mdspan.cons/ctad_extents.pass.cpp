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

// No CTAD in C++14 or earlier
//UNSUPPORTED: c++14

#include <hip/std/mdspan>
#include <hip/std/cassert>

#define CHECK_MDSPAN(m,d) \
    static_assert(m.is_exhaustive(), ""); \
    assert(m.data_handle()  == d.data()); \
    assert(m.rank()         == 2       ); \
    assert(m.rank_dynamic() == 2       ); \
    assert(m.extent(0)      == 64      ); \
    assert(m.extent(1)      == 128     )


int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
    // TEST(TestMdspanCTAD, extents_object)
    {
        hip::std::array<int, 1> d{42};
        hip::std::mdspan m{d.data(), hip::std::extents{64, 128}};

        CHECK_MDSPAN(m,d);
    }

    // TEST(TestMdspanCTAD, extents_object_move)
    {
        hip::std::array<int, 1> d{42};
        hip::std::mdspan m{d.data(), std::move(hip::std::extents{64, 128})};

        CHECK_MDSPAN(m,d);
    }

    // TEST(TestMdspanCTAD, extents_std_array)
    {
        hip::std::array<int, 1> d{42};
        hip::std::mdspan m{d.data(), hip::std::array{64, 128}};

        CHECK_MDSPAN(m,d);
    }

    // TEST(TestMdspanCTAD, cptr_extents_std_array)
    {
        hip::std::array<int, 1> d{42};
        const int* const ptr= d.data();
        hip::std::mdspan m{ptr, hip::std::array{64, 128}};

        static_assert(hip::std::is_same<typename decltype(m)::element_type, const int>::value, "");

        CHECK_MDSPAN(m,d);
    }

    // extents from std::span
    {
        hip::std::array<int, 1> d{42};
        hip::std::array<int, 2> sarr{64, 128};
        hip::std::mdspan m{d.data(), hip::std::span{sarr}};

        CHECK_MDSPAN(m,d);
    }
#endif

    return 0;
}
