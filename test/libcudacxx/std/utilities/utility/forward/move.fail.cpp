//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// This test should pass in C++03 with Clang extensions because Clang does
// not implicitly delete the copy constructor when move constructors are
// defaulted using extensions.

// XFAIL: c++03

// test move

#include "test_macros.h"
#include <cuda/std/cassert>
#include <cuda/std/utility>

struct move_only
{
  TEST_FUNC move_only() {}
  move_only(move_only&&)            = default;
  move_only& operator=(move_only&&) = default;
};

TEST_FUNC move_only source()
{
  return move_only();
}
TEST_FUNC const move_only csource()
{
  return move_only();
}

TEST_FUNC void test(move_only) {}

int main(int, char**)
{
  const move_only ca = move_only();
  // expected-error@+1 {{call to implicitly-deleted copy constructor of 'move_only'}}
  test(std::move(ca));

  return 0;
}
