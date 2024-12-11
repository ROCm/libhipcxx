//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// <cuda/std/iterator>

// template <class T, class charT = char, class traits = char_traits<charT>,
//           class Distance = ptrdiff_t>
// class istream_iterator
//     : public iterator<input_iterator_tag, T, Distance, const T*, const T&>
// {
// public:
//     typedef charT char_type;
//     typedef traits traits_type;
//     typedef basic_istream<charT,traits> istream_type;
//     ...
//
// Before C++17, we have:
//   If T is a literal type, then the default constructor shall be a constexpr constructor.
//   If T is a literal type, then this constructor shall be a trivial copy constructor.
//   If T is a literal type, then this destructor shall be a trivial destructor.
// C++17 says:
//   If is_trivially_default_constructible_v<T> is true, then
//       this constructor (the default ctor) is a constexpr constructor.
//   If is_trivially_copy_constructible_v<T> is true, then
//       this constructor (the copy ctor) is a trivial copy constructor.
//   If is_trivially_destructible_v<T> is true, then this
//       destructor is a trivial destructor.
//  Testing the C++17 ctors for this are in the ctor tests.

#include <hip/std/iterator>
#include <hip/std/type_traits>
#if defined(_LIBCUDACXX_HAS_STRING)
#include <hip/std/string>

#include "test_macros.h"

int main(int, char**)
{
    typedef hip::std::istream_iterator<double> I1; // double is trivially destructible
#if TEST_STD_VER <= 14
    static_assert((hip::std::is_convertible<I1,
        hip::std::iterator<hip::std::input_iterator_tag, double, hip::std::ptrdiff_t,
        const double*, const double&> >::value), "");
#else
    static_assert((hip::std::is_same<I1::iterator_category, hip::std::input_iterator_tag>::value), "");
    static_assert((hip::std::is_same<I1::value_type, double>::value), "");
    static_assert((hip::std::is_same<I1::difference_type, hip::std::ptrdiff_t>::value), "");
    static_assert((hip::std::is_same<I1::pointer, const double*>::value), "");
    static_assert((hip::std::is_same<I1::reference, const double&>::value), "");
#endif
    static_assert((hip::std::is_same<I1::char_type, char>::value), "");
    static_assert((hip::std::is_same<I1::traits_type, hip::std::char_traits<char> >::value), "");
    static_assert((hip::std::is_same<I1::istream_type, hip::std::istream>::value), "");
    static_assert( hip::std::is_trivially_copy_constructible<I1>::value, "");
    static_assert( hip::std::is_trivially_destructible<I1>::value, "");

    typedef hip::std::istream_iterator<unsigned, wchar_t> I2; // unsigned is trivially destructible
#if TEST_STD_VER <= 14
    static_assert((hip::std::is_convertible<I2,
        hip::std::iterator<hip::std::input_iterator_tag, unsigned, hip::std::ptrdiff_t,
        const unsigned*, const unsigned&> >::value), "");
#else
    static_assert((hip::std::is_same<I2::iterator_category, hip::std::input_iterator_tag>::value), "");
    static_assert((hip::std::is_same<I2::value_type, unsigned>::value), "");
    static_assert((hip::std::is_same<I2::difference_type, hip::std::ptrdiff_t>::value), "");
    static_assert((hip::std::is_same<I2::pointer, const unsigned*>::value), "");
    static_assert((hip::std::is_same<I2::reference, const unsigned&>::value), "");
#endif
    static_assert((hip::std::is_same<I2::char_type, wchar_t>::value), "");
    static_assert((hip::std::is_same<I2::traits_type, hip::std::char_traits<wchar_t> >::value), "");
    static_assert((hip::std::is_same<I2::istream_type, hip::std::wistream>::value), "");
    static_assert( hip::std::is_trivially_copy_constructible<I2>::value, "");
    static_assert( hip::std::is_trivially_destructible<I2>::value, "");

    typedef hip::std::istream_iterator<hip::std::string> I3; // string is NOT trivially destructible
    static_assert(!hip::std::is_trivially_copy_constructible<I3>::value, "");
    static_assert(!hip::std::is_trivially_destructible<I3>::value, "");

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
