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

__host__ __device__ void test_std_swap_static_extents()
{
    int data1[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int data2[12] = {21,22,23,24,25,26,27,28,29,30,31,32};

    hip::std::mdspan<int, hip::std::extents<size_t,3,4>> m1(data1);
    hip::std::mdspan<int, hip::std::extents<size_t,3,4>> m2(data2);
    hip::std::extents<size_t,3,4> exts1;
    hip::std::layout_right::mapping<hip::std::extents<size_t, 3, 4>> map1(exts1);
    hip::std::extents<size_t,3,4> exts2;
    hip::std::layout_right::mapping<hip::std::extents<size_t, 3, 4>> map2(exts2);

    assert(m1.data_handle() == data1);
    assert(m1.mapping() == map1);
    auto val1 = m1(0,0);
    assert(val1 == 1);
    assert(m2.data_handle() == data2);
    assert(m2.mapping() == map2);
    auto val2 = m2(0,0);
    assert(val2 == 21);

    hip::std::swap(m1,m2);
    assert(m1.data_handle() == data2);
    assert(m1.mapping() == map2);
    val1 = m1(0,0);
    assert(val1 == 21);
    assert(m2.data_handle() == data1);
    assert(m2.mapping() == map1);
    val2 = m2(0,0);
    assert(val2 == 1);
}

__host__ __device__ void test_std_swap_dynamic_extents()
{
    int data1[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int data2[12] = {21,22,23,24,25,26,27,28,29,30,31,32};

    hip::std::mdspan<int, hip::std::dextents<size_t,2>> m1(data1,3,4);
    hip::std::mdspan<int, hip::std::dextents<size_t,2>> m2(data2,4,3);
    hip::std::dextents<size_t,2> exts1(3,4);
    hip::std::layout_right::mapping<hip::std::dextents<size_t,2>> map1(exts1);
    hip::std::dextents<size_t,2> exts2(4,3);
    hip::std::layout_right::mapping<hip::std::dextents<size_t,2>> map2(exts2);

    assert(m1.data_handle() == data1);
    assert(m1.mapping() == map1);
    auto val1 = m1(0,0);
    assert(val1 == 1);
    assert(m2.data_handle() == data2);
    assert(m2.mapping() == map2);
    auto val2 = m2(0,0);
    assert(val2 == 21);

    hip::std::swap(m1,m2);
    assert(m1.data_handle() == data2);
    assert(m1.mapping() == map2);
    val1 = m1(0,0);
    assert(val1 == 21);
    assert(m2.data_handle() == data1);
    assert(m2.mapping() == map1);
    val2 = m2(0,0);
    assert(val2 == 1);
}

int main(int, char**)
{
    test_std_swap_static_extents();

    test_std_swap_dynamic_extents();

    //TODO port tests for customized layout and accessor

    return 0;
}
