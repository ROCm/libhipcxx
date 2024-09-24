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

// XFAIL: gcc-4
// UNSUPPORTED: nvrtc

// <utility>

// template <class T1, class T2> struct pair

// explicit(see-below) constexpr pair();

// NOTE: The SFINAE on the default constructor is tested in
//       default-sfinae.pass.cpp


#include <hip/std/utility>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "archetypes.h"

int main(int, char**)
{
    {
        typedef hip::std::pair<float, short*> P;
        P p;
        assert(p.first == 0.0f);
        assert(p.second == nullptr);
    }
#if TEST_STD_VER >= 11
    {
        typedef hip::std::pair<float, short*> P;
        constexpr P p;
        static_assert(p.first == 0.0f, "");
        static_assert(p.second == nullptr, "");
    }
    {
        using NoDefault = ImplicitTypes::NoDefault;
        using P = hip::std::pair<int, NoDefault>;
        static_assert(!hip::std::is_default_constructible<P>::value, "");
        using P2 = hip::std::pair<NoDefault, int>;
        static_assert(!hip::std::is_default_constructible<P2>::value, "");
    }
    {
        struct Base { };
        struct Derived : Base { protected: Derived() = default; };
        static_assert(!hip::std::is_default_constructible<hip::std::pair<Derived, int> >::value, "");
    }
#endif

  return 0;
}
