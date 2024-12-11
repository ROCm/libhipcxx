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
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/utility>
// hip::std::unique_ptr not supported
//#include <hip/std/memory>
// hip::std::string not supported
//#include <hip/std/string>
// hip::std::complex not supported
//#include <hip/std/complex>
#include <hip/std/type_traits>

#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    // hip::std::complex not supported
    // hip::std::string not supported
    /*
    typedef hip::std::complex<float> cf;
    {
    auto t1 = hip::std::tuple<int, hip::std::string, cf> { 42, "Hi", { 1,2 }};
    assert ( hip::std::get<int>(t1) == 42 ); // find at the beginning
    assert ( hip::std::get<hip::std::string>(t1) == "Hi" ); // find in the middle
    assert ( hip::std::get<cf>(t1).real() == 1 ); // find at the end
    assert ( hip::std::get<cf>(t1).imag() == 2 );
    }

    {
    auto t2 = hip::std::tuple<int, hip::std::string, int, cf> { 42, "Hi", 23, { 1,2 }};
//  get<int> would fail!
    assert ( hip::std::get<hip::std::string>(t2) == "Hi" );
    assert (( hip::std::get<cf>(t2) == cf{ 1,2 } ));
    }
    */
    {
    constexpr hip::std::tuple<int, const int, double, double> p5 { 1, 2, 3.4, 5.6 };
    static_assert ( hip::std::get<int>(p5) == 1, "" );
    static_assert ( hip::std::get<const int>(p5) == 2, "" );
    }

    {
    const hip::std::tuple<int, const int, double, double> p5 { 1, 2, 3.4, 5.6 };
    const int &i1 = hip::std::get<int>(p5);
    const int &i2 = hip::std::get<const int>(p5);
    assert ( i1 == 1 );
    assert ( i2 == 2 );
    }

    // hip::std::unique_ptr not supported
    /*
    {
    typedef hip::std::unique_ptr<int> upint;
    hip::std::tuple<upint> t(upint(new int(4)));
    upint p = hip::std::get<upint>(hip::std::move(t)); // get rvalue
    assert(*p == 4);
    assert(hip::std::get<upint>(t) == nullptr); // has been moved from
    }

    {
    typedef hip::std::unique_ptr<int> upint;
    const hip::std::tuple<upint> t(upint(new int(4)));
    const upint&& p = hip::std::get<upint>(hip::std::move(t)); // get const rvalue
    assert(*p == 4);
    assert(hip::std::get<upint>(t) != nullptr);
    }
    */

    {
    int x = 42;
    int y = 43;
    hip::std::tuple<int&, int const&> const t(x, y);
    static_assert(hip::std::is_same<int&, decltype(hip::std::get<int&>(hip::std::move(t)))>::value, "");
    static_assert(noexcept(hip::std::get<int&>(hip::std::move(t))), "");
    static_assert(hip::std::is_same<int const&, decltype(hip::std::get<int const&>(hip::std::move(t)))>::value, "");
    static_assert(noexcept(hip::std::get<int const&>(hip::std::move(t))), "");
    }

#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
    int x = 42;
    int y = 43;
    hip::std::tuple<int&&, int const&&> const t(hip::std::move(x), hip::std::move(y));
    static_assert(hip::std::is_same<int&&, decltype(hip::std::get<int&&>(hip::std::move(t)))>::value, "");
    static_assert(noexcept(hip::std::get<int&&>(hip::std::move(t))), "");
    static_assert(hip::std::is_same<int const&&, decltype(hip::std::get<int const&&>(hip::std::move(t)))>::value, "");
    static_assert(noexcept(hip::std::get<int const&&>(hip::std::move(t))), "");
    }
#endif
    {
    constexpr const hip::std::tuple<int, const int, double, double> t { 1, 2, 3.4, 5.6 };
    static_assert(hip::std::get<int>(hip::std::move(t)) == 1, "");
    static_assert(hip::std::get<const int>(hip::std::move(t)) == 2, "");
    }

  return 0;
}
