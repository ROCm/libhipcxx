// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libhipcxx, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Copyright (c) 2024 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: nvhpc, nvc++

// include twice to make sure that there are no header conflicts
#include <hip/std/limits>
#include <cuda/std/limits>

#include <hip/atomic>
#include <cuda/std/atomic>

int main(int, char**)
{
  // Test exactly matching definitions that can be used in hip and cuda namespace
  static_assert(hip::std::numeric_limits<int>::max() == cuda::std::numeric_limits<int>::max());

  // #include <hip*> header, but use cuda namespace
  cuda::atomic<int> a1;

  // #include <cuda*> header, but use hip namespace
  hip::std::atomic<int> a2;

  return 0;
}
