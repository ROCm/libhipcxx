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

#include <hip/std/mdspan>
#include <hip/std/tuple>

#define TEST_TYPE \
    hip::std::tuple < \
        hip::std::extents<size_t, Extents...>, \
        hip::std::integer_sequence<size_t, DynamicSizes...> \
    >


template <class> struct TestExtents;
template <size_t... Extents, size_t... DynamicSizes>
struct TestExtents<
    hip::std::tuple<
        hip::std::extents<size_t, Extents...>,
        hip::std::integer_sequence<size_t, DynamicSizes...>
    >
>
{
    using extents_type = hip::std::extents<size_t,Extents...>;
    // Double Braces here to make it work with GCC 5
    // Otherwise: "error: array must be initialized with a brace-enclosed initializer"
    const hip::std::array<size_t, sizeof...(Extents)> static_sizes {{ Extents... }};
    const hip::std::array<size_t, sizeof...(DynamicSizes)> dyn_sizes {{ DynamicSizes... }};
    extents_type exts { DynamicSizes... };
};

template <size_t... Ds>
using _sizes = hip::std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using _exts = hip::std::extents<size_t,Ds...>;

constexpr auto dyn = hip::std::dynamic_extent;

using extents_test_types = hip::std::tuple<
    hip::std::tuple< _exts< 10     >, _sizes<     > >
  , hip::std::tuple< _exts<dyn     >, _sizes<10   > >
  , hip::std::tuple< _exts< 10,   3>, _sizes<     > >
  , hip::std::tuple< _exts<dyn,   3>, _sizes<10   > >
  , hip::std::tuple< _exts< 10, dyn>, _sizes< 3   > >
  , hip::std::tuple< _exts<dyn, dyn>, _sizes<10, 3> >
>;
