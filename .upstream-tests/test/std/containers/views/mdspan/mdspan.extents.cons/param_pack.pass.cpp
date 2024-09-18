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

__host__ __device__ void check( hip::std::dextents<size_t,2> e )
{
    static_assert( e.rank        () == 2, "" );
    static_assert( e.rank_dynamic() == 2, "" );

    assert( e.extent(0) == 2 );
    assert( e.extent(1) == 2 );
}

template< class, class T, class... IndexTypes >
struct is_param_pack_cons_avail : hip::std::false_type {};

template< class T, class... IndexTypes >
struct is_param_pack_cons_avail< hip::std::enable_if_t< hip::std::is_same< decltype( T{ hip::std::declval<IndexTypes>()... } )
                                                                           , T
                                                                           >::value
                                                       >
                               , T
                               , IndexTypes...
                               > : hip::std::true_type {};

template< class T, class... IndexTypes >
constexpr bool is_param_pack_cons_avail_v = is_param_pack_cons_avail< void, T, IndexTypes... >::value;

int main(int, char**)
{
    {
        hip::std::dextents<int,2> e{2, 2};

        check( e );
    }

    {
        hip::std::dextents<int,2> e(2, 2);

        check( e );
    }

#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
    {
        hip::std::extents e{2, 2};

        check( e );
    }

    {
        hip::std::extents e(2, 2);

        check( e );
    }
#endif

    {
        hip::std::dextents<size_t,2> e{2, 2};

        check( e );
    }

    static_assert( is_param_pack_cons_avail_v< hip::std::dextents<int,2>, int   , int    > == true , "" );

    static_assert( is_param_pack_cons_avail_v< hip::std::dextents<int,2>, my_int, my_int > == true , "" );

    // Constraint: rank consistency
    static_assert( is_param_pack_cons_avail_v< hip::std::dextents<int,1>, int   , int    > == false, "" );

    // Constraint: convertibility
    static_assert( is_param_pack_cons_avail_v< hip::std::dextents<int,1>, my_int_non_convertible           > == false, "" );

    // Constraint: nonthrow-constructibility
    static_assert( is_param_pack_cons_avail_v< hip::std::dextents<int,1>, my_int_non_nothrow_constructible > == false, "" );

    return 0;
}

