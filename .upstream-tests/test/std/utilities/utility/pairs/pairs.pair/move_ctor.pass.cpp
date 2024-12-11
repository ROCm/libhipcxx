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

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// pair(pair&&) = default;

#include <hip/std/utility>
// cuda/std/memory not supported
// #include <hip/std/memory>
#include <hip/std/cassert>

#include "test_macros.h"

struct Dummy {
  Dummy(Dummy const&) = delete;
  Dummy(Dummy &&) = default;
};

int main(int, char**)
{
    {
        typedef hip::std::pair<int, short> P1;
        static_assert(hip::std::is_move_constructible<P1>::value, "");
        P1 p1(3, static_cast<short>(4));
        P1 p2 = hip::std::move(p1);
        assert(p2.first == 3);
        assert(p2.second == 4);
    }
    {
        using P = hip::std::pair<Dummy, int>;
        static_assert(!hip::std::is_copy_constructible<P>::value, "");
        static_assert(hip::std::is_move_constructible<P>::value, "");
    }

  return 0;
}
