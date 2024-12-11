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

// UNSUPPORTED: c++98, c++03, c++11
// UNSUPPORTED: msvc

#include <hip/std/utility>
// hip::std::string not supported
// #include <hip/std/string>
#include <hip/std/type_traits>
// cuda/std/complex not supported
// #include <hip/std/complex>
// cuda/std/memory not supported
// #include <hip/std/memory>

#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    // cuda/std/complex not supported
    /*
    typedef hip::std::complex<float> cf;
    {
    auto t1 = hip::std::make_pair<int, cf> ( 42, { 1,2 } );
    assert ( hip::std::get<int>(t1) == 42 );
    assert ( hip::std::get<cf>(t1).real() == 1 );
    assert ( hip::std::get<cf>(t1).imag() == 2 );
    }
    */
    {
    const hip::std::pair<int, const int> p1 { 1, 2 };
    const int &i1 = hip::std::get<int>(p1);
    const int &i2 = hip::std::get<const int>(p1);
    assert ( i1 == 1 );
    assert ( i2 == 2 );
    }

    // cuda/std/memory not supported
    /*
    {
    typedef hip::std::unique_ptr<int> upint;
    hip::std::pair<upint, int> t(upint(new int(4)), 42);
    upint p = hip::std::get<upint>(hip::std::move(t)); // get rvalue
    assert(*p == 4);
    assert(hip::std::get<upint>(t) == nullptr); // has been moved from
    }

    {
    typedef hip::std::unique_ptr<int> upint;
    const hip::std::pair<upint, int> t(upint(new int(4)), 42);
    static_assert(hip::std::is_same<const upint&&, decltype(hip::std::get<upint>(hip::std::move(t)))>::value, "");
    static_assert(noexcept(hip::std::get<upint>(hip::std::move(t))), "");
    static_assert(hip::std::is_same<const int&&, decltype(hip::std::get<int>(hip::std::move(t)))>::value, "");
    static_assert(noexcept(hip::std::get<int>(hip::std::move(t))), "");
    auto&& p = hip::std::get<upint>(hip::std::move(t)); // get const rvalue
    auto&& i = hip::std::get<int>(hip::std::move(t)); // get const rvalue
    assert(*p == 4);
    assert(i == 42);
    assert(hip::std::get<upint>(t) != nullptr);
    }
    */

    {
    int x = 42;
    int const y = 43;
    hip::std::pair<int&, int const&> const p(x, y);
    static_assert(hip::std::is_same<int&, decltype(hip::std::get<int&>(hip::std::move(p)))>::value, "");
    static_assert(noexcept(hip::std::get<int&>(hip::std::move(p))), "");
    static_assert(hip::std::is_same<int const&, decltype(hip::std::get<int const&>(hip::std::move(p)))>::value, "");
    static_assert(noexcept(hip::std::get<int const&>(hip::std::move(p))), "");
    }

    {
    int x = 42;
    int const y = 43;
    hip::std::pair<int&&, int const&&> const p(hip::std::move(x), hip::std::move(y));
    static_assert(hip::std::is_same<int&&, decltype(hip::std::get<int&&>(hip::std::move(p)))>::value, "");
    static_assert(noexcept(hip::std::get<int&&>(hip::std::move(p))), "");
    static_assert(hip::std::is_same<int const&&, decltype(hip::std::get<int const&&>(hip::std::move(p)))>::value, "");
    static_assert(noexcept(hip::std::get<int const&&>(hip::std::move(p))), "");
    }

    {
    constexpr const hip::std::pair<int, const int> p { 1, 2 };
    static_assert(hip::std::get<int>(hip::std::move(p)) == 1, "");
    static_assert(hip::std::get<const int>(hip::std::move(p)) == 2, "");
    }

  return 0;
}
