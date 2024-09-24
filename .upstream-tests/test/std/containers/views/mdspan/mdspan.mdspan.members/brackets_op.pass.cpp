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

// Will be testing `m[0,0]` when it becomes available
// Alternatively, could use macro `__MDSPAN_OP(m,0,0)` which is turned to either `m[0,0]` or `m(0,0)`,
// depending on if `__cpp_multidimensional_subscript` is defined or not

constexpr auto dyn = hip::std::dynamic_extent;

template< class, class T, class... OtherIndexTypes >
struct is_bracket_op_avail : hip::std::false_type {};

template< class T, class... OtherIndexTypes >
struct is_bracket_op_avail< hip::std::enable_if_t< hip::std::is_same< decltype( hip::std::declval<T>()(hip::std::declval<OtherIndexTypes>()...) )
                                                                      , typename T::accessor_type::reference
                                                                      >::value
                                                  >
                          , T
                          , OtherIndexTypes...
                          > : hip::std::true_type {};

template< class T, class... OtherIndexTypes >
constexpr bool is_bracket_op_avail_v = is_bracket_op_avail< void, T, OtherIndexTypes... >::value;

template< class T, class OtherIndexType, size_t N, class = void >
struct is_bracket_op_array_avail : hip::std::false_type {};

template< class T, class OtherIndexType, size_t N >
struct is_bracket_op_array_avail< T
                                , OtherIndexType
                                , N
                                , hip::std::enable_if_t< hip::std::is_same< decltype( hip::std::declval<T>()(hip::std::declval<hip::std::array<OtherIndexType,N>>()) )
                                                                            , typename T::accessor_type::reference
                                                                            >::value
                                                        >
                                > : hip::std::true_type {};

template< class T, class OtherIndexType, size_t N >
constexpr bool is_bracket_op_array_avail_v = is_bracket_op_array_avail< T, OtherIndexType, N >::value;

template< class T, class OtherIndexType, size_t N, class = void >
struct is_bracket_op_span_avail : hip::std::false_type {};

template< class T, class OtherIndexType, size_t N >
struct is_bracket_op_span_avail< T
                               , OtherIndexType
                               , N
                               , hip::std::enable_if_t< hip::std::is_same< decltype( hip::std::declval<T>()(hip::std::declval<hip::std::span<OtherIndexType,N>>()) )
                                                                           , typename T::accessor_type::reference
                                                                           >::value
                                                       >
                               > : hip::std::true_type {};

template< class T, class OtherIndexType, size_t N >
constexpr bool is_bracket_op_span_avail_v = is_bracket_op_span_avail< T, OtherIndexType, N >::value;


int main(int, char**)
{
    {
        using element_t = int;
        using   index_t = int;
        using     ext_t = hip::std::extents<index_t, dyn, dyn>;
        using  mdspan_t = hip::std::mdspan<element_t, ext_t>;

        hip::std::array<element_t, 4> d{42,43,44,45};
        mdspan_t m{d.data(), ext_t{2, 2}};

        static_assert( is_bracket_op_avail_v< decltype(m), int, int > == true, "" );

        // param pack
        assert( m(0,0) == 42 );
        assert( m(0,1) == 43 );
        assert( m(1,0) == 44 );
        assert( m(1,1) == 45 );

        // array of indices
        assert( m(hip::std::array<int,2>{0,0}) == 42 );
        assert( m(hip::std::array<int,2>{0,1}) == 43 );
        assert( m(hip::std::array<int,2>{1,0}) == 44 );
        assert( m(hip::std::array<int,2>{1,1}) == 45 );

        static_assert( is_bracket_op_array_avail_v< decltype(m), int, 2 > == true, "" );

        // span of indices
        assert( m(hip::std::span<const int,2>{hip::std::array<int,2>{0,0}}) == 42 );
        assert( m(hip::std::span<const int,2>{hip::std::array<int,2>{0,1}}) == 43 );
        assert( m(hip::std::span<const int,2>{hip::std::array<int,2>{1,0}}) == 44 );
        assert( m(hip::std::span<const int,2>{hip::std::array<int,2>{1,1}}) == 45 );

        static_assert( is_bracket_op_span_avail_v< decltype(m), int, 2 > == true, "" );
    }

    // Param pack of indices in a type implicitly convertible to index_type
    {
        using element_t = int;
        using   index_t = int;
        using     ext_t = hip::std::extents<index_t, dyn, dyn>;
        using  mdspan_t = hip::std::mdspan<element_t, ext_t>;

        hip::std::array<element_t, 4> d{42,43,44,45};
        mdspan_t m{d.data(), ext_t{2, 2}};

        assert( m(my_int(0),my_int(0)) == 42 );
        assert( m(my_int(0),my_int(1)) == 43 );
        assert( m(my_int(1),my_int(0)) == 44 );
        assert( m(my_int(1),my_int(1)) == 45 );
    }

    // Constraint: rank consistency
    {
        using element_t = int;
        using   index_t = int;
        using  mdspan_t = hip::std::mdspan<element_t, hip::std::extents<index_t,dyn>>;

        static_assert( is_bracket_op_avail_v< mdspan_t, index_t, index_t > == false, "" );

        static_assert( is_bracket_op_array_avail_v< mdspan_t, index_t, 2 > == false, "" );

        static_assert( is_bracket_op_span_avail_v < mdspan_t, index_t, 2 > == false, "" );
    }

    // Constraint: convertibility
    {
        using element_t = int;
        using   index_t = int;
        using  mdspan_t = hip::std::mdspan<element_t, hip::std::extents<index_t,dyn>>;

        static_assert( is_bracket_op_avail_v< mdspan_t, my_int_non_convertible > == false, "" );

        static_assert( is_bracket_op_array_avail_v< mdspan_t, my_int_non_convertible, 1 > == false, "" );

        static_assert( is_bracket_op_span_avail_v < mdspan_t, my_int_non_convertible, 1 > == false, "" );
    }

    // Constraint: nonthrow-constructibility
    {
        using element_t = int;
        using   index_t = int;
        using  mdspan_t = hip::std::mdspan<element_t, hip::std::extents<index_t,dyn>>;

        static_assert( is_bracket_op_avail_v< mdspan_t, my_int_non_nothrow_constructible > == false, "" );

        static_assert( is_bracket_op_array_avail_v< mdspan_t, my_int_non_nothrow_constructible, 1 > == false, "" );

        static_assert( is_bracket_op_span_avail_v < mdspan_t, my_int_non_nothrow_constructible, 1 > == false, "" );
    }

    return 0;
}
