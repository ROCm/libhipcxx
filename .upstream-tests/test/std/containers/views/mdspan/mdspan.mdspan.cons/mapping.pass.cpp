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
#include "../my_accessor.hpp"

constexpr auto dyn = hip::std::dynamic_extent;

template< class T, class DataHandleType, class MappingType, class = void >
struct is_mapping_cons_avail : hip::std::false_type {};

template< class T, class DataHandleType, class MappingType >
struct is_mapping_cons_avail< T
                            , DataHandleType
                            , MappingType
                            , hip::std::enable_if_t< hip::std::is_same< decltype( T{ hip::std::declval<DataHandleType>()
                                                                                     , hip::std::declval<MappingType>()
                                                                                     }
                                                                                  )
                                                                        , T
                                                                        >::value
                                                    >
                            > : hip::std::true_type {};

template< class T, class DataHandleType, class MappingType >
constexpr bool is_mapping_cons_avail_v = is_mapping_cons_avail< T, DataHandleType, MappingType >::value;

int main(int, char**)
{
    using    data_t = int;
    using   index_t = size_t;
    using     ext_t = hip::std::extents<index_t,dyn,dyn>;
    using mapping_t = hip::std::layout_left::mapping<ext_t>;

    // mapping
    {
        using  mdspan_t = hip::std::mdspan<data_t, ext_t, hip::std::layout_left>;

        static_assert( is_mapping_cons_avail_v< mdspan_t, int *, mapping_t > == true, "" );

        hip::std::array<data_t, 1> d{42};
        mapping_t map{hip::std::dextents<index_t,2>{64, 128}};
        mdspan_t  m{ d.data(), map };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // Constraint: is_default_constructible_v<accessor_type> is true
    {
        using  mdspan_t = hip::std::mdspan<data_t, ext_t, hip::std::layout_left, Foo::my_accessor<data_t>>;

        static_assert( is_mapping_cons_avail_v< mdspan_t, int *, mapping_t > == false, "" );
    }

    // mapping and accessor
    {

        hip::std::array<data_t, 1> d{42};
        mapping_t map{hip::std::dextents<index_t,2>{64, 128}};
        hip::std::default_accessor<data_t> a;
        hip::std::mdspan<data_t, ext_t, hip::std::layout_left> m{ d.data(), map, a };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    return 0;
}
