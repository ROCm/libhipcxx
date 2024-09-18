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

// aligned_storage
//
//  Issue 3034 added:
//  The member typedef type shall be a trivial standard-layout type.

#include <hip/std/type_traits>
#include <hip/std/cstddef>       // for hip::std::max_align_t
#include "test_macros.h"

int main(int, char**)
{
    {
    typedef hip::std::aligned_storage<10, 1 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<10, 1>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef hip::std::aligned_storage<10, 2 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<10, 2>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef hip::std::aligned_storage<10, 4 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<10, 4>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 12, "");
    }
    {
    typedef hip::std::aligned_storage<10, 8 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<10, 8>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef hip::std::aligned_storage<10, 16 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<10, 16>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef hip::std::aligned_storage<10, 32 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<10, 32>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
    }
    {
    typedef hip::std::aligned_storage<20, 32 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<20, 32>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
    }
    {
    typedef hip::std::aligned_storage<40, 32 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<40, 32>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 64, "");
    }
    {
    typedef hip::std::aligned_storage<12, 16 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<12, 16>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef hip::std::aligned_storage<1>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<1>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 1, "");
    }
    {
    typedef hip::std::aligned_storage<2>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<2>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 2, "");
    }
    {
    typedef hip::std::aligned_storage<3>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<3>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 4, "");
    }
    {
    typedef hip::std::aligned_storage<4>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<4>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 4, "");
    }
    {
    typedef hip::std::aligned_storage<5>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<5>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
    }
    {
    typedef hip::std::aligned_storage<7>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<7>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
    }
    {
    typedef hip::std::aligned_storage<8>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<8>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 8, "");
    }
    {
    typedef hip::std::aligned_storage<9>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<9>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef hip::std::aligned_storage<15>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<15>);
#endif
#if TEST_STD_VER <= 17
    static_assert(hip::std::is_pod<T1>::value, "");
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    // Use alignof(hip::std::max_align_t) below to find the max alignment instead of
    // hardcoding it, because it's different on different platforms.
    // (For example 8 on arm and 16 on x86.)
    {
    typedef hip::std::aligned_storage<16>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<16>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == TEST_ALIGNOF(hip::std::max_align_t),
                  "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef hip::std::aligned_storage<17>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<17>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == TEST_ALIGNOF(hip::std::max_align_t),
                  "");
    static_assert(sizeof(T1) == 16 + TEST_ALIGNOF(hip::std::max_align_t), "");
    }
    {
    typedef hip::std::aligned_storage<10>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, hip::std::aligned_storage_t<10>);
#endif
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
// NVCC doesn't support types that are _this_ overaligned, it seems
#if !defined(TEST_COMPILER_NVCC) && !defined(TEST_COMPILER_NVRTC)
  {
    const int Align = 65536;
    typedef typename hip::std::aligned_storage<1, Align>::type T1;
    static_assert(hip::std::is_trivial<T1>::value, "");
    static_assert(hip::std::is_standard_layout<T1>::value, "");
    static_assert(hip::std::alignment_of<T1>::value == Align, "");
    static_assert(sizeof(T1) == Align, "");
  }
#endif

  return 0;
}
