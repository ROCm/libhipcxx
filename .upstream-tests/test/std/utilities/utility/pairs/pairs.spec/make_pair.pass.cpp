//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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

// <utility>

// template <class T1, class T2> pair<V1, V2> make_pair(T1&&, T2&&);

#include <hip/std/utility>
// cuda/std/memory not supported
// #include <hip/std/memory>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef hip::std::pair<int, short> P1;
        P1 p1 = hip::std::make_pair(3, static_cast<short>(4));
        assert(p1.first == 3);
        assert(p1.second == 4);
    }

#if TEST_STD_VER >= 11
    // cuda/std/memory not supported
    /*
    {
        typedef hip::std::pair<hip::std::unique_ptr<int>, short> P1;
        P1 p1 = hip::std::make_pair(hip::std::unique_ptr<int>(new int(3)), static_cast<short>(4));
        assert(*p1.first == 3);
        assert(p1.second == 4);
    }
    {
        typedef hip::std::pair<hip::std::unique_ptr<int>, short> P1;
        P1 p1 = hip::std::make_pair(nullptr, static_cast<short>(4));
        assert(p1.first == nullptr);
        assert(p1.second == 4);
    }
    */
#endif
#if TEST_STD_VER >= 14
    {
        typedef hip::std::pair<int, short> P1;
        constexpr P1 p1 = hip::std::make_pair(3, static_cast<short>(4));
        static_assert(p1.first == 3, "");
        static_assert(p1.second == 4, "");
    }
#endif


  return 0;
}
