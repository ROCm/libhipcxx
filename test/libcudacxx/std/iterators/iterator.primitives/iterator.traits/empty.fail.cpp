//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

// struct iterator_traits
// {
// };

#include <cuda/std/iterator>

#include "test_macros.h"

struct A
{};
struct NotAnIteratorEmpty
{};

struct NotAnIteratorNoDifference
{
  //     using difference_type = int;
  using value_type        = A;
  using pointer           = A*;
  using reference         = A&;
  using iterator_category = cuda::std::forward_iterator_tag;
};

struct NotAnIteratorNoValue
{
  using difference_type = int;
  //     using value_type = A;
  using pointer           = A*;
  using reference         = A&;
  using iterator_category = cuda::std::forward_iterator_tag;
};

struct NotAnIteratorNoPointer
{
  using difference_type = int;
  using value_type      = A;
  //     using pointer = A*;
  using reference         = A&;
  using iterator_category = cuda::std::forward_iterator_tag;
};

struct NotAnIteratorNoReference
{
  using difference_type = int;
  using value_type      = A;
  using pointer         = A*;
  //    using reference = A&;
  using iterator_category = cuda::std::forward_iterator_tag;
};

struct NotAnIteratorNoCategory
{
  using difference_type = int;
  using value_type      = A;
  using pointer         = A*;
  using reference       = A&;
  //     using iterator_category = cuda::std::forward_iterator_tag;
};

int main(int, char**)
{
  {
    using T  = cuda::std::iterator_traits<NotAnIteratorEmpty>;
    using DT = T::difference_type; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using VT = T::value_type; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using PT = T::pointer; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using RT = T::reference; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using CT = T::iterator_category; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  {
    using T  = cuda::std::iterator_traits<NotAnIteratorNoDifference>;
    using DT = T::difference_type; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using VT = T::value_type; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using PT = T::pointer; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using RT = T::reference; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using CT = T::iterator_category; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  {
    using T  = cuda::std::iterator_traits<NotAnIteratorNoValue>;
    using DT = T::difference_type; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using VT = T::value_type; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using PT = T::pointer; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using RT = T::reference; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using CT = T::iterator_category; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }
#if TEST_STD_VER <= 2017
  {
    using T  = cuda::std::iterator_traits<NotAnIteratorNoPointer>;
    using DT = T::difference_type; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using VT = T::value_type; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using PT = T::pointer; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using RT = T::reference; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using CT = T::iterator_category; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }
#endif
  {
    using T  = cuda::std::iterator_traits<NotAnIteratorNoReference>;
    using DT = T::difference_type; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using VT = T::value_type; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using PT = T::pointer; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using RT = T::reference; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using CT = T::iterator_category; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  {
    using T  = cuda::std::iterator_traits<NotAnIteratorNoCategory>;
    using DT = T::difference_type; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using VT = T::value_type; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using PT = T::pointer; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using RT = T::reference; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    using CT = T::iterator_category; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  return 0;
}
