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
// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: nvrtc

// This test also generates spurious warnings when instantiating std::span
// with a very large extent (like size_t(-2)) -- silence those.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=warning

// <span>

// template<size_t Offset, size_t Count = dynamic_extent>
//   constexpr span<element_type, see below> subspan() const;
//
//  Requires: offset <= size() &&
//            (count == dynamic_extent || count <= size() - offset)

#include <hip/std/span>
#include <hip/std/cstddef>

void f() {
  int array[] = {1, 2, 3, 4};
  hip::std::span<const int, 4> sp(array);

  //  Offset too large templatized
  [[maybe_unused]] auto s1 = sp.subspan<5>(); // expected-error@span:* {{span<T, N>::subspan<Offset, Count>(): Offset out of range}}

  //  Count too large templatized
  [[maybe_unused]] auto s2 = sp.subspan<0, 5>(); // expected-error@span:* {{span<T, N>::subspan<Offset, Count>(): Offset + Count out of range}}

  //  Offset + Count too large templatized
  [[maybe_unused]] auto s3 = sp.subspan<2, 3>(); // expected-error@span:* {{span<T, N>::subspan<Offset, Count>(): Offset + Count out of range}}

  //  Offset + Count overflow templatized
  [[maybe_unused]] auto s4 = sp.subspan<3, hip::std::size_t(-2)>(); // expected-error@span:* {{span<T, N>::subspan<Offset, Count>(): Offset + Count out of range}}, expected-error-re@span:* {{array is too large{{(.* elements)}}}}
}
