//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// type_traits

// aligned_union<size_t Len, class ...Types>

//  Issue 3034 added:
//  The member typedef type shall be a trivial standard-layout type.

#include <hip/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef hip::std::aligned_union<10, char >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_union_t<10, char>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef hip::std::aligned_union<10, short >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_union_t<10, short>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef hip::std::aligned_union<10, int >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_union_t<10, int>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 12, "");
    }
    {
    typedef hip::std::aligned_union<10, double >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_union_t<10, double>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef hip::std::aligned_union<10, short, char >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_union_t<10, short, char>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef hip::std::aligned_union<10, char, short >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_union_t<10, char, short>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef hip::std::aligned_union<2, int, char, short >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_union_t<2, int, char, short>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 4, "");
    }
    {
    typedef hip::std::aligned_union<2, char, int, short >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_union_t<2, char, int, short>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 4, "");
    }
    {
    typedef hip::std::aligned_union<2, char, short, int >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_union_t<2, char, short, int>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 4, "");
    }

  return 0;
}
