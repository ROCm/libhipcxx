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

// template<class charT, class traits = char_traits<charT> >
// class istreambuf_iterator
//     : public iterator<input_iterator_tag, charT,
//                       typename traits::off_type, unspecified,
//                       charT>
// {
// public:
//     typedef charT                         char_type;
//     typedef traits                        traits_type;
//     typedef typename traits::int_type     int_type;
//     typedef basic_streambuf<charT,traits> streambuf_type;
//     typedef basic_istream<charT,traits>   istream_type;
//     ...
//
// All specializations of istreambuf_iterator shall have a trivial copy constructor,
//    a constexpr default constructor and a trivial destructor.

#include <hip/std/iterator>
#if defined(_LIBCUDACXX_HAS_STRING)
#include <hip/std/string>
#include <hip/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
    typedef hip::std::istreambuf_iterator<char> I1;
    static_assert((hip::std::is_same<I1::iterator_category, hip::std::input_iterator_tag>::value), "");
    static_assert((hip::std::is_same<I1::value_type, char>::value), "");
    static_assert((hip::std::is_same<I1::difference_type, hip::std::char_traits<char>::off_type>::value), "");
    LIBCPP_STATIC_ASSERT((hip::std::is_same<I1::pointer, char*>::value), "");
    static_assert((hip::std::is_same<I1::reference, char>::value), "");
    static_assert((hip::std::is_same<I1::char_type, char>::value), "");
    static_assert((hip::std::is_same<I1::traits_type, hip::std::char_traits<char> >::value), "");
    static_assert((hip::std::is_same<I1::int_type, I1::traits_type::int_type>::value), "");
    static_assert((hip::std::is_same<I1::streambuf_type, hip::std::streambuf>::value), "");
    static_assert((hip::std::is_same<I1::istream_type, hip::std::istream>::value), "");
    static_assert((hip::std::is_nothrow_default_constructible<I1>::value), "" );
    static_assert((hip::std::is_trivially_copy_constructible<I1>::value), "" );
    static_assert((hip::std::is_trivially_destructible<I1>::value), "" );

    typedef hip::std::istreambuf_iterator<wchar_t> I2;
    static_assert((hip::std::is_same<I2::iterator_category, hip::std::input_iterator_tag>::value), "");
    static_assert((hip::std::is_same<I2::value_type, wchar_t>::value), "");
    static_assert((hip::std::is_same<I2::difference_type, hip::std::char_traits<wchar_t>::off_type>::value), "");
    LIBCPP_STATIC_ASSERT((hip::std::is_same<I2::pointer, wchar_t*>::value), "");
    static_assert((hip::std::is_same<I2::reference, wchar_t>::value), "");
    static_assert((hip::std::is_same<I2::char_type, wchar_t>::value), "");
    static_assert((hip::std::is_same<I2::traits_type, hip::std::char_traits<wchar_t> >::value), "");
    static_assert((hip::std::is_same<I2::int_type, I2::traits_type::int_type>::value), "");
    static_assert((hip::std::is_same<I2::streambuf_type, hip::std::wstreambuf>::value), "");
    static_assert((hip::std::is_same<I2::istream_type, hip::std::wistream>::value), "");
    static_assert((hip::std::is_nothrow_default_constructible<I2>::value), "" );
    static_assert((hip::std::is_trivially_copy_constructible<I2>::value), "" );
    static_assert((hip::std::is_trivially_destructible<I2>::value), "" );

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
