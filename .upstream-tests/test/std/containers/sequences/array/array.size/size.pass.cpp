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

// template <class T, size_t N> constexpr size_type array<T,N>::size();

#include <hip/std/array>
#include <hip/std/cassert>

#include "test_macros.h"

// hip::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{
    {
        typedef double T;
        typedef hip::std::array<T, 3> C;
        C c = {1, 2, 3.5};
        assert(c.size() == 3);
        assert(c.max_size() == 3);
        assert(!c.empty());
    }
    {
        typedef double T;
        typedef hip::std::array<T, 0> C;
        C c = {};
        assert(c.size() == 0);
        assert(c.max_size() == 0);
        assert(c.empty());
    }
#if TEST_STD_VER >= 11
    {
        typedef double T;
        typedef hip::std::array<T, 3> C;
        constexpr C c = {1, 2, 3.5};
        static_assert(c.size() == 3, "");
        static_assert(c.max_size() == 3, "");
        static_assert(!c.empty(), "");
    }
    {
        typedef double T;
        typedef hip::std::array<T, 0> C;
        constexpr C c = {};
        static_assert(c.size() == 0, "");
        static_assert(c.max_size() == 0, "");
        static_assert(c.empty(), "");
    }
#endif

  return 0;
}
