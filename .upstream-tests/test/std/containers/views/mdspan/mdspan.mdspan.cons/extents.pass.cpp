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
#include "../my_int.hpp"
#include "../my_accessor.hpp"

constexpr auto dyn = hip::std::dynamic_extent;

template< class T, class DataHandleType, class ExtentsType, class = void >
struct is_extents_cons_avail : hip::std::false_type {};

template< class T, class DataHandleType, class ExtentsType >
struct is_extents_cons_avail< T
                            , DataHandleType
                            , ExtentsType
                            , hip::std::enable_if_t< hip::std::is_same< decltype( T{ hip::std::declval<DataHandleType>()
                                                                                     , hip::std::declval<ExtentsType>()
                                                                                     }
                                                                                  )
                                                                        , T
                                                                        >::value
                                                    >
                            > : hip::std::true_type {};

template< class T, class DataHandleType, class ExtentsType >
constexpr bool is_extents_cons_avail_v = is_extents_cons_avail< T, DataHandleType, ExtentsType >::value;

int main(int, char**)
{
    // extents from extents object
    {
        using ext_t = hip::std::extents<int, dyn, dyn>;
        hip::std::array<int, 1> d{42};
        hip::std::mdspan<int, ext_t> m{ d.data(), ext_t{64, 128} };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // extents from extents object move
    {
        using    ext_t = hip::std::extents< int, dyn, dyn >;
        using mdspan_t = hip::std::mdspan< int, ext_t >;

        static_assert( is_extents_cons_avail_v< mdspan_t, int *, ext_t > == true, "" );

        hip::std::array<int, 1> d{42};
        mdspan_t m{ d.data(), hip::std::move(ext_t{64, 128}) };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // Constraint: is_constructible_v<mapping_type, extents_type> is true
    {
        using    ext_t = hip::std::extents< int, 16, 16 >;
        using mdspan_t = hip::std::mdspan< int, ext_t, hip::std::layout_stride >;

        static_assert( is_extents_cons_avail_v< mdspan_t, int *, ext_t > == false, "" );
    }

    // Constraint: is_default_constructible_v<accessor_type> is true
    {
        using    ext_t = hip::std::extents< int, 16, 16 >;
        using mdspan_t = hip::std::mdspan< int, ext_t, hip::std::layout_right, Foo::my_accessor<int> >;

        static_assert( is_extents_cons_avail_v< mdspan_t, int *, ext_t > == false, "" );
    }

    return 0;
}

