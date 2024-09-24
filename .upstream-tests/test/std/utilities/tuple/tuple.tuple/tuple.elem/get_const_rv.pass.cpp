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



// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
//   const typename tuple_element<I, tuple<Types...> >::type&&
//   get(const tuple<Types...>&& t);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/utility>
// hip::std::string not supported
//#include <hip/std/string>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef hip::std::tuple<int> T;
    const T t(3);
    static_assert(hip::std::is_same<const int&&, decltype(hip::std::get<0>(hip::std::move(t)))>::value, "");
    static_assert(noexcept(hip::std::get<0>(hip::std::move(t))), "");
    const int&& i = hip::std::get<0>(hip::std::move(t));
    assert(i == 3);
    }

    // hip::std::string not supported
    /*
    {
    typedef hip::std::tuple<hip::std::string, int> T;
    const T t("high", 5);
    static_assert(hip::std::is_same<const hip::std::string&&, decltype(hip::std::get<0>(hip::std::move(t)))>::value, "");
    static_assert(noexcept(hip::std::get<0>(hip::std::move(t))), "");
    static_assert(hip::std::is_same<const int&&, decltype(hip::std::get<1>(hip::std::move(t)))>::value, "");
    static_assert(noexcept(hip::std::get<1>(hip::std::move(t))), "");
    const hip::std::string&& s = hip::std::get<0>(hip::std::move(t));
    const int&& i = hip::std::get<1>(hip::std::move(t));
    assert(s == "high");
    assert(i == 5);
    }
    */

    {
    int x = 42;
    int const y = 43;
    hip::std::tuple<int&, int const&> const p(x, y);
    static_assert(hip::std::is_same<int&, decltype(hip::std::get<0>(hip::std::move(p)))>::value, "");
    static_assert(noexcept(hip::std::get<0>(hip::std::move(p))), "");
    static_assert(hip::std::is_same<int const&, decltype(hip::std::get<1>(hip::std::move(p)))>::value, "");
    static_assert(noexcept(hip::std::get<1>(hip::std::move(p))), "");
    }

#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
    int x = 42;
    int const y = 43;
    hip::std::tuple<int&&, int const&&> const p(hip::std::move(x), hip::std::move(y));
    static_assert(hip::std::is_same<int&&, decltype(hip::std::get<0>(hip::std::move(p)))>::value, "");
    static_assert(noexcept(hip::std::get<0>(hip::std::move(p))), "");
    static_assert(hip::std::is_same<int const&&, decltype(hip::std::get<1>(hip::std::move(p)))>::value, "");
    static_assert(noexcept(hip::std::get<1>(hip::std::move(p))), "");
    }
#endif

#if TEST_STD_VER > 11
    {
    typedef hip::std::tuple<double, int> T;
    constexpr const T t(2.718, 5);
    static_assert(hip::std::get<0>(hip::std::move(t)) == 2.718, "");
    static_assert(hip::std::get<1>(hip::std::move(t)) == 5, "");
    }
#endif

  return 0;
}
