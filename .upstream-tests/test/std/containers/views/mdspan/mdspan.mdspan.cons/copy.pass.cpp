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

template< class T1, class T0, class = void >
struct is_copy_cons_avail : hip::std::false_type {};

template< class T1, class T0 >
struct is_copy_cons_avail< T1
                         , T0
                         , hip::std::enable_if_t< hip::std::is_same< decltype( T1{ hip::std::declval<T0>() } ), T1 >::value >
                         > : hip::std::true_type {};

template< class T1, class T0 >
constexpr bool is_copy_cons_avail_v = is_copy_cons_avail< T1, T0 >::value;

int main(int, char**)
{
    // copy constructor
    {
        using ext_t    = hip::std::extents<int, dyn, dyn>;
        using mdspan_t = hip::std::mdspan<int, ext_t>;

        static_assert( is_copy_cons_avail_v< mdspan_t, mdspan_t > == true, "" );

        hip::std::array<int, 1> d{42};
        mdspan_t m0{ d.data(), ext_t{64, 128} };
        mdspan_t m { m0 };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // copy constructor with conversion
    {
        hip::std::array<int, 1> d{42};
        hip::std::mdspan<int, hip::std::extents<int,64,128>> m0{ d.data(), hip::std::extents<int,64,128>{} };
        hip::std::mdspan<const int, hip::std::extents<size_t,dyn,dyn>> m { m0 };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // Constraint: is_constructible_v<mapping_type, const OtherLayoutPolicy::template mapping<OtherExtents>&> is true
    {
        using mdspan1_t = hip::std::mdspan<int, hip::std::extents<int,dyn,dyn>, hip::std::layout_left >;
        using mdspan0_t = hip::std::mdspan<int, hip::std::extents<int,dyn,dyn>, hip::std::layout_right>;

        static_assert( is_copy_cons_avail_v< mdspan1_t, mdspan0_t > == false, "" );
    }

    // Constraint: is_constructible_v<accessor_type, const OtherAccessor&> is true
    {
        using mdspan1_t = hip::std::mdspan<int, hip::std::extents<int,dyn,dyn>, hip::std::layout_right, Foo::my_accessor<int>>;
        using mdspan0_t = hip::std::mdspan<int, hip::std::extents<int,dyn,dyn>, hip::std::layout_right>;

        static_assert( is_copy_cons_avail_v< mdspan1_t, mdspan0_t > == false, "" );
    }

    return 0;
}
