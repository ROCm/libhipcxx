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

// class ostreambuf_iterator

// bool failed() const throw();

#include <hip/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#include <hip/std/sstream>
#include <hip/std/cassert>

#include "test_macros.h"

template <typename Char, typename Traits = hip::std::char_traits<Char> >
struct my_streambuf : public hip::std::basic_streambuf<Char,Traits> {
    typedef typename hip::std::basic_streambuf<Char,Traits>::int_type  int_type;
    typedef typename hip::std::basic_streambuf<Char,Traits>::char_type char_type;

    my_streambuf() {}
    int_type sputc(char_type) { return Traits::eof(); }
    };

int main(int, char**)
{
    {
        my_streambuf<char> buf;
        hip::std::ostreambuf_iterator<char> i(&buf);
        i = 'a';
        assert(i.failed());
    }
    {
        my_streambuf<wchar_t> buf;
        hip::std::ostreambuf_iterator<wchar_t> i(&buf);
        i = L'a';
        assert(i.failed());
    }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
