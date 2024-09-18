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

// istreambuf_iterator

// bool equal(istreambuf_iterator<charT,traits>& b) const;

#include <hip/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#include <hip/std/sstream>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        hip::std::istringstream inf1("abc");
        hip::std::istringstream inf2("def");
        hip::std::istreambuf_iterator<char> i1(inf1);
        hip::std::istreambuf_iterator<char> i2(inf2);
        hip::std::istreambuf_iterator<char> i3;
        hip::std::istreambuf_iterator<char> i4;
        hip::std::istreambuf_iterator<char> i5(nullptr);

        assert( i1.equal(i1));
        assert( i1.equal(i2));
        assert(!i1.equal(i3));
        assert(!i1.equal(i4));
        assert(!i1.equal(i5));

        assert( i2.equal(i1));
        assert( i2.equal(i2));
        assert(!i2.equal(i3));
        assert(!i2.equal(i4));
        assert(!i2.equal(i5));

        assert(!i3.equal(i1));
        assert(!i3.equal(i2));
        assert( i3.equal(i3));
        assert( i3.equal(i4));
        assert( i3.equal(i5));

        assert(!i4.equal(i1));
        assert(!i4.equal(i2));
        assert( i4.equal(i3));
        assert( i4.equal(i4));
        assert( i4.equal(i5));

        assert(!i5.equal(i1));
        assert(!i5.equal(i2));
        assert( i5.equal(i3));
        assert( i5.equal(i4));
        assert( i5.equal(i5));
    }
    {
        hip::std::wistringstream inf1(L"abc");
        hip::std::wistringstream inf2(L"def");
        hip::std::istreambuf_iterator<wchar_t> i1(inf1);
        hip::std::istreambuf_iterator<wchar_t> i2(inf2);
        hip::std::istreambuf_iterator<wchar_t> i3;
        hip::std::istreambuf_iterator<wchar_t> i4;
        hip::std::istreambuf_iterator<wchar_t> i5(nullptr);

        assert( i1.equal(i1));
        assert( i1.equal(i2));
        assert(!i1.equal(i3));
        assert(!i1.equal(i4));
        assert(!i1.equal(i5));

        assert( i2.equal(i1));
        assert( i2.equal(i2));
        assert(!i2.equal(i3));
        assert(!i2.equal(i4));
        assert(!i2.equal(i5));

        assert(!i3.equal(i1));
        assert(!i3.equal(i2));
        assert( i3.equal(i3));
        assert( i3.equal(i4));
        assert( i3.equal(i5));

        assert(!i4.equal(i1));
        assert(!i4.equal(i2));
        assert( i4.equal(i3));
        assert( i4.equal(i4));
        assert( i4.equal(i5));

        assert(!i5.equal(i1));
        assert(!i5.equal(i2));
        assert( i5.equal(i3));
        assert( i5.equal(i4));
        assert( i5.equal(i5));
    }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
