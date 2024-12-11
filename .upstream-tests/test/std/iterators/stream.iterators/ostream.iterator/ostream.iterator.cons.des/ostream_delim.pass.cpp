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

// class ostream_iterator

// ostream_iterator(ostream_type& s, const charT* delimiter);

#include <hip/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#include <hip/std/sstream>
#include <hip/std/cassert>

#include "test_macros.h"


struct MyTraits : hip::std::char_traits<char> {};

typedef hip::std::basic_ostringstream<char, MyTraits> StringStream;
typedef hip::std::basic_ostream<char, MyTraits> BasicStream;

void operator&(BasicStream const&) {}

int main(int, char**)
{
    {
        hip::std::ostringstream outf;
        hip::std::ostream_iterator<int> i(outf, ", ");
        assert(outf.good());
    }
    {
        hip::std::wostringstream outf;
        hip::std::ostream_iterator<double, wchar_t> i(outf, L", ");
        assert(outf.good());
    }
    {
        StringStream outf;
        hip::std::ostream_iterator<int, char, MyTraits> i(outf, ", ");
        assert(outf.good());
    }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
