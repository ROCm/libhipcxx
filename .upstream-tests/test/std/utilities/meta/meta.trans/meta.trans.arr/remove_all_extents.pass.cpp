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

// remove_all_extents

#include <hip/std/type_traits>

#include "test_macros.h"

enum Enum {zero, one_};

template <class T, class U>
__host__ __device__
void test_remove_all_extents()
{
    ASSERT_SAME_TYPE(U, typename hip::std::remove_all_extents<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U,        hip::std::remove_all_extents_t<T>);
#endif
}

int main(int, char**)
{
    test_remove_all_extents<int, int> ();
    test_remove_all_extents<const Enum, const Enum> ();
    test_remove_all_extents<int[], int> ();
    test_remove_all_extents<const int[], const int> ();
    test_remove_all_extents<int[3], int> ();
    test_remove_all_extents<const int[3], const int> ();
    test_remove_all_extents<int[][3], int> ();
    test_remove_all_extents<const int[][3], const int> ();
    test_remove_all_extents<int[2][3], int> ();
    test_remove_all_extents<const int[2][3], const int> ();
    test_remove_all_extents<int[1][2][3], int> ();
    test_remove_all_extents<const int[1][2][3], const int> ();

  return 0;
}
