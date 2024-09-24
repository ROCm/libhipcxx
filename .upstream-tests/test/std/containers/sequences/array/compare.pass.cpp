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

// <cuda/std/array>

//  These are all constexpr in C++20
// bool operator==(array<T, N> const&, array<T, N> const&);
// bool operator!=(array<T, N> const&, array<T, N> const&);
// bool operator<(array<T, N> const&, array<T, N> const&);
// bool operator<=(array<T, N> const&, array<T, N> const&);
// bool operator>(array<T, N> const&, array<T, N> const&);
// bool operator>=(array<T, N> const&, array<T, N> const&);


#include <hip/std/array>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#include <hip/std/vector>
#endif
#include <hip/std/cassert>

#include "test_macros.h"
#include "test_comparisons.h"

// hip::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{
  {
    typedef int T;
    typedef hip::std::array<T, 3> C;
    C c1 = {1, 2, 3};
    C c2 = {1, 2, 3};
    C c3 = {3, 2, 1};
    C c4 = {1, 2, 1};
    assert(testComparisons6(c1, c2, true, false));
    assert(testComparisons6(c1, c3, false, true));
    assert(testComparisons6(c1, c4, false, false));
  }
  {
    typedef int T;
    typedef hip::std::array<T, 0> C;
    C c1 = {};
    C c2 = {};
    assert(testComparisons6(c1, c2, true, false));
  }

#if TEST_STD_VER > 17
  {
  constexpr hip::std::array<int, 3> a1 = {1, 2, 3};
  constexpr hip::std::array<int, 3> a2 = {2, 3, 4};
  static_assert(testComparisons6(a1, a1, true, false), "");
  static_assert(testComparisons6(a1, a2, false, true), "");
  static_assert(testComparisons6(a2, a1, false, false), "");
  }
#endif

  return 0;
}
