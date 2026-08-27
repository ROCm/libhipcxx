//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: nvcc, nvhpc, nvc++

// <cuda/barrier>

// Verify hip::aligned_size_t<N> construction, traits, and value conversion.

#include <cuda/barrier>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

template<hip::std::size_t Align, hip::std::size_t TestValue>
__host__ __device__ constexpr bool test_aligned_size()
{
  using aligned_t = hip::aligned_size_t<Align>;

  static_assert(!hip::std::is_default_constructible<aligned_t>::value, "");
  static_assert(aligned_t::align == Align, "");

  if (hip::std::is_default_constructible<aligned_t>::value)
    return false;
  if (aligned_t::align != Align)
    return false;

  const aligned_t aligned{TestValue};
  if (aligned.value != TestValue)
    return false;
  if (static_cast<hip::std::size_t>(aligned) != TestValue)
    return false;

  return true;
}

__host__ __device__ constexpr bool test()
{
  if (!test_aligned_size<1, 42>())
    return false;

  if (!test_aligned_size<16, 128>())
    return false;

  if (!test_aligned_size<128, 256>())
    return false;

  return true;
}

static_assert(hip::aligned_size_t<42>{1337}.value == 1337, "");

int main(int, char**)
{
  bool result = test();
  return result ? 0 : 1;
}
