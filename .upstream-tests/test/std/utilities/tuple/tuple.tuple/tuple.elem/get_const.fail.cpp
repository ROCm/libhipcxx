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
//   typename tuple_element<I, tuple<Types...> >::type const&
//   get(const tuple<Types...>& t);

// UNSUPPORTED: c++98, c++03 
// UNSUPPORTED: nvrtc

#include <hip/std/tuple>
// hip::std::string not supported
//#include <hip/std/string>
#include <hip/std/cassert>

int main(int, char**)
{
    // hip::std::string not supported
    /*
    {
        typedef hip::std::tuple<double&, hip::std::string, int> T;
        double d = 1.5;
        const T t(d, "high", 5);
        assert(hip::std::get<0>(t) == 1.5);
        assert(hip::std::get<1>(t) == "high");
        assert(hip::std::get<2>(t) == 5);
        hip::std::get<0>(t) = 2.5;
        assert(hip::std::get<0>(t) == 2.5);
        assert(hip::std::get<1>(t) == "high");
        assert(hip::std::get<2>(t) == 5);
        assert(d == 2.5);

        hip::std::get<1>(t) = "four";
    }
    */
    {
        typedef hip::std::tuple<double&, int> T;
        double d = 1.5;
        const T t(d, 5);
        assert(hip::std::get<0>(t) == 1.5);
        assert(hip::std::get<1>(t) == 5);
        hip::std::get<0>(t) = 2.5;
        assert(hip::std::get<0>(t) == 2.5);
        assert(hip::std::get<1>(t) == 5);
        assert(d == 2.5);

        // Expected failure: <1> is not a modifiable lvalue
        hip::std::get<1>(t) = 10;
    }
  return 0;
}