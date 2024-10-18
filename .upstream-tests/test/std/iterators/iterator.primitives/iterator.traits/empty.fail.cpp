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

// UNSUPPORTED: nvrtc

// struct iterator_traits
// {
// };

#include <hip/std/iterator>
#include "test_macros.h"

struct A {};
struct NotAnIteratorEmpty {};

struct NotAnIteratorNoDifference
{
//     typedef int                       difference_type;
    typedef A                         value_type;
    typedef A*                        pointer;
    typedef A&                        reference;
    typedef hip::std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoValue
{
    typedef int                       difference_type;
//     typedef A                         value_type;
    typedef A*                        pointer;
    typedef A&                        reference;
    typedef hip::std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoPointer
{
    typedef int                       difference_type;
    typedef A                         value_type;
//     typedef A*                        pointer;
    typedef A&                        reference;
    typedef hip::std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoReference
{
    typedef int                       difference_type;
    typedef A                         value_type;
    typedef A*                        pointer;
//    typedef A&                        reference;
    typedef hip::std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoCategory
{
    typedef int                       difference_type;
    typedef A                         value_type;
    typedef A*                        pointer;
    typedef A&                        reference;
//     typedef hip::std::forward_iterator_tag iterator_category;
};

int main(int, char**)
{
    {
    typedef hip::std::iterator_traits<NotAnIteratorEmpty> T;
    typedef T::difference_type   DT; // expected-error-re {{no type named 'difference_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::value_type        VT; // expected-error-re {{no type named 'value_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::pointer           PT; // expected-error-re {{no type named 'pointer' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::reference         RT; // expected-error-re {{no type named 'reference' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in 'hip::std::iterator_traits<{{.+}}>}}
    }

    {
    typedef hip::std::iterator_traits<NotAnIteratorNoDifference> T;
    typedef T::difference_type   DT; // expected-error-re {{no type named 'difference_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::value_type        VT; // expected-error-re {{no type named 'value_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::pointer           PT; // expected-error-re {{no type named 'pointer' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::reference         RT; // expected-error-re {{no type named 'reference' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in 'hip::std::iterator_traits<{{.+}}>}}
    }

    {
    typedef hip::std::iterator_traits<NotAnIteratorNoValue> T;
    typedef T::difference_type   DT; // expected-error-re {{no type named 'difference_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::value_type        VT; // expected-error-re {{no type named 'value_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::pointer           PT; // expected-error-re {{no type named 'pointer' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::reference         RT; // expected-error-re {{no type named 'reference' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in 'hip::std::iterator_traits<{{.+}}>}}
    }

    {
    typedef hip::std::iterator_traits<NotAnIteratorNoPointer> T;
    typedef T::difference_type   DT; // expected-error-re {{no type named 'difference_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::value_type        VT; // expected-error-re {{no type named 'value_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::pointer           PT; // expected-error-re {{no type named 'pointer' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::reference         RT; // expected-error-re {{no type named 'reference' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in 'hip::std::iterator_traits<{{.+}}>}}
    }

    {
    typedef hip::std::iterator_traits<NotAnIteratorNoReference> T;
    typedef T::difference_type   DT; // expected-error-re {{no type named 'difference_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::value_type        VT; // expected-error-re {{no type named 'value_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::pointer           PT; // expected-error-re {{no type named 'pointer' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::reference         RT; // expected-error-re {{no type named 'reference' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in 'hip::std::iterator_traits<{{.+}}>}}
    }

    {
    typedef hip::std::iterator_traits<NotAnIteratorNoCategory> T;
    typedef T::difference_type   DT; // expected-error-re {{no type named 'difference_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::value_type        VT; // expected-error-re {{no type named 'value_type' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::pointer           PT; // expected-error-re {{no type named 'pointer' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::reference         RT; // expected-error-re {{no type named 'reference' in 'hip::std::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in 'hip::std::iterator_traits<{{.+}}>}}
    }

  return 0;
}
